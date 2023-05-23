#include <ginkgo/ginkgo.hpp>
#include __NCCL_INC
#include <benchmark/benchmark.h>


template<typename F>
double timed(F &&fn) {
    auto start = std::chrono::steady_clock::now();

    fn();

    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<
            std::chrono::duration<double>>(end - start).count();
}


// returns positive values even for negative a
int mod(int a, int b) {
    return (a % b + b) % b;
}


class AllGather : public benchmark::Fixture {
public:
    void SetUp(benchmark::State &state) override {
        const auto size = static_cast<gko::size_type >(state.range(0));

        send_buf = gko::array<double>(exec, size);
        recv_buf = gko::array<double>(exec, size * mpi_comm.size());
    }

    gko::experimental::mpi::communicator mpi_comm = MPI_COMM_WORLD;
    ncclComm_t nccl_comm;

    std::shared_ptr<gko::__EXEC> exec;

    gko::array<double> send_buf;
    gko::array<double> recv_buf;
};

class AllReduce : public benchmark::Fixture {
public:
    void SetUp(benchmark::State &state) override {
        const auto size = static_cast<gko::size_type >(state.range(0));

        send_buf = gko::array<double>(exec, size);
        recv_buf = gko::array<double>(exec, size);
    }

    gko::experimental::mpi::communicator mpi_comm = MPI_COMM_WORLD;
    ncclComm_t nccl_comm;

    std::shared_ptr<gko::__EXEC> exec;

    gko::array<double> send_buf;
    gko::array<double> recv_buf;
};

class AllToAll : public benchmark::Fixture {
public:
    void SetUp(benchmark::State &state) override {
        const auto size = static_cast<gko::size_type >(state.range(0));

        send_sizes.resize(mpi_comm.size(), 0);
        recv_sizes.resize(mpi_comm.size(), 0);
        send_offsets.resize(mpi_comm.size() + 1);
        recv_offsets.resize(mpi_comm.size() + 1);

        auto comm_size = mpi_comm.size();
        auto rank = mpi_comm.rank();
        auto offset = static_cast<int>(std::sqrt(comm_size));
        auto r_x = rank % offset;
        auto r_y = rank / offset;

        auto target_rank = [&](auto ix, auto iy) {
            return ix + iy * offset;
        };

        send_sizes[target_rank(mod(r_x + 1, offset), r_y)] += size;
        send_sizes[target_rank(mod(r_x - 1, offset), r_y)] += size;
        send_sizes[target_rank(r_x, mod(r_y + 1, offset))] += size;
        send_sizes[target_rank(r_x, mod(r_y - 1, offset))] += size;

        recv_sizes[target_rank(mod(r_x + 1, offset), r_y)] += size;
        recv_sizes[target_rank(mod(r_x - 1, offset), r_y)] += size;
        recv_sizes[target_rank(r_x, mod(r_y + 1, offset))] += size;
        recv_sizes[target_rank(r_x, mod(r_y - 1, offset))] += size;

        std::partial_sum(send_sizes.begin(), send_sizes.end(), send_offsets.begin() + 1);
        std::partial_sum(recv_sizes.begin(), recv_sizes.end(), recv_offsets.begin() + 1);


        send_buf = gko::array<double>(exec, send_offsets.back());
        recv_buf = gko::array<double>(exec, recv_offsets.back());


        auto source = mpi_comm.rank();
        // compute degree for own rank
        auto degree = static_cast<int>(
                std::count_if(send_sizes.begin(), send_sizes.end(),
                              [](const auto v) { return v > 0; }));
        // destinations are ranks with send_size > 0
        std::vector<int> destinations;
        std::vector<int> weight;
        for (int r = 0; r < send_sizes.size(); ++r) {
            if (send_sizes[r] > 0) {
                destinations.push_back(r);
                weight.push_back(send_sizes[r]);
            }
        }

        MPI_Comm graph;
        MPI_Dist_graph_create(mpi_comm.get(), 1, &source, &degree, destinations.data(),
                              weight.data(), MPI_INFO_NULL, true, &graph);

        int num_in_neighbors;
        int num_out_neighbors;
        int weighted;
        MPI_Dist_graph_neighbors_count(graph, &num_in_neighbors, &num_out_neighbors,
                                       &weighted);

        std::vector<int> out_neighbors(num_out_neighbors);
        std::vector<int> in_neighbors(num_in_neighbors);
        std::vector<int> out_weight(num_out_neighbors);
        std::vector<int> in_weight(num_in_neighbors);
        MPI_Dist_graph_neighbors(graph, num_in_neighbors, in_neighbors.data(),
                                 in_weight.data(), num_out_neighbors,
                                 out_neighbors.data(), out_weight.data());

        neighbor_comm = gko::experimental::mpi::communicator{graph};

        // compress communication info
        neighbor_recv_offsets.resize(num_in_neighbors);
        neighbor_send_offsets.resize(num_out_neighbors);
        for (int r = 0; r < in_neighbors.size(); ++r) {
            neighbor_recv_offsets[r] = recv_offsets[in_neighbors[r]];
        }
        neighbor_recv_offsets.back() = recv_offsets.back();
        for (int r = 0; r < out_neighbors.size(); ++r) {
            neighbor_send_offsets[r] = send_offsets[out_neighbors[r]];
        }
        neighbor_send_offsets.back() = send_offsets.back();
        neighbor_send_sizes = std::move(out_weight);
        neighbor_recv_sizes = std::move(in_weight);

        reqs.reserve(num_in_neighbors + num_out_neighbors);
    }

    gko::experimental::mpi::communicator mpi_comm = MPI_COMM_WORLD;
    gko::experimental::mpi::communicator neighbor_comm = MPI_COMM_WORLD;
    ncclComm_t nccl_comm;

    std::shared_ptr<gko::__EXEC> exec;

    gko::array<double> send_buf;
    gko::array<double> recv_buf;

    std::vector<int> send_sizes;
    std::vector<int> send_offsets;
    std::vector<int> recv_sizes;
    std::vector<int> recv_offsets;

    std::vector<int> neighbor_send_sizes;
    std::vector<int> neighbor_send_offsets;
    std::vector<int> neighbor_recv_sizes;
    std::vector<int> neighbor_recv_offsets;

    std::vector<gko::experimental::mpi::request> reqs;
};


BENCHMARK_DEFINE_F(AllGather, None)(benchmark::State &state) {
    const auto num_kernels = state.range(1);

    for (auto _: state) {
        auto start = std::chrono::steady_clock::now();
        auto elapsed_seconds = timed(
                [&] {
                    for (int i = 0; i < num_kernels; ++i) {
                        send_buf.fill(mpi_comm.rank());
                        benchmark::DoNotOptimize(send_buf.get_data());
                    }

                    exec->synchronize();
                }
        );
        mpi_comm.all_reduce(exec->get_master(), &elapsed_seconds, 1, MPI_MAX);
        state.SetIterationTime(elapsed_seconds);
    }
}

BENCHMARK_DEFINE_F(AllGather, MPI)(benchmark::State &state) {
    const auto num_kernels = state.range(1);

    for (auto _: state) {
        auto elapsed_seconds = timed(
                [&] {
                    for (int i = 0; i < num_kernels; ++i) {
                        send_buf.fill(mpi_comm.rank());
                        benchmark::DoNotOptimize(send_buf.get_data());
                    }

                    exec->synchronize();
                    mpi_comm.all_gather(exec, send_buf.get_data(), send_buf.get_num_elems(), recv_buf.get_data(),
                                        send_buf.get_num_elems());
                    exec->synchronize();
                }
        );
        mpi_comm.all_reduce(exec->get_master(), &elapsed_seconds, 1, MPI_MAX);
        state.SetIterationTime(elapsed_seconds);
    }
}

BENCHMARK_DEFINE_F(AllGather, NCCL)(benchmark::State &state) {
    const auto num_kernels = state.range(1);

    for (auto _: state) {
        auto elapsed_seconds = timed(
                [&] {
                    for (int i = 0; i < num_kernels; ++i) {
                        send_buf.fill(mpi_comm.rank());
                        benchmark::DoNotOptimize(send_buf.get_data());
                    }

                    {
                        auto g = exec->get_scoped_device_id_guard();
                        ncclAllGather(send_buf.get_data(), recv_buf.get_data(), send_buf.get_num_elems(), ncclDouble,
                                      nccl_comm, exec->get_stream());
                    }

                    exec->synchronize();
                }
        );
        mpi_comm.all_reduce(exec->get_master(), &elapsed_seconds, 1, MPI_MAX);
        state.SetIterationTime(elapsed_seconds);
    }
}


BENCHMARK_DEFINE_F(AllReduce, MPI)(benchmark::State &state) {
    const auto num_kernels = state.range(1);

    for (auto _: state) {
        auto elapsed_seconds = timed(
                [&] {
                    for (int i = 0; i < num_kernels; ++i) {
                        send_buf.fill(mpi_comm.rank());
                        benchmark::DoNotOptimize(send_buf.get_data());
                    }

                    exec->synchronize();
                    mpi_comm.all_reduce(exec, send_buf.get_data(), recv_buf.get_data(),
                                        send_buf.get_num_elems(), MPI_SUM);
                    exec->synchronize();
                }
        );
        mpi_comm.all_reduce(exec->get_master(), &elapsed_seconds, 1, MPI_MAX);
        state.SetIterationTime(elapsed_seconds);
    }
}

BENCHMARK_DEFINE_F(AllReduce, NCCL)(benchmark::State &state) {
    const auto num_kernels = state.range(1);

    for (auto _: state) {
        auto elapsed_seconds = timed(
                [&] {
                    for (int i = 0; i < num_kernels; ++i) {
                        send_buf.fill(mpi_comm.rank());
                        benchmark::DoNotOptimize(send_buf.get_data());
                    }

                    {
                        auto g = exec->get_scoped_device_id_guard();
                        ncclAllReduce(send_buf.get_data(), recv_buf.get_data(), send_buf.get_num_elems(), ncclDouble,
                                      ncclSum, nccl_comm, exec->get_stream());
                    }

                    exec->synchronize();
                }
        );
        mpi_comm.all_reduce(exec->get_master(), &elapsed_seconds, 1, MPI_MAX);
        state.SetIterationTime(elapsed_seconds);
    }
}


BENCHMARK_DEFINE_F(AllToAll, MPI)(benchmark::State &state) {
    const auto num_kernels = state.range(1);

    for (auto _: state) {
        auto elapsed_seconds = timed(
                [&] {
                    for (int i = 0; i < num_kernels; ++i) {
                        send_buf.fill(mpi_comm.rank());
                        benchmark::DoNotOptimize(send_buf.get_data());
                    }

                    exec->synchronize();
                    mpi_comm.all_to_all_v(exec, send_buf.get_data(), send_sizes.data(), send_offsets.data(),
                                          recv_buf.get_data(),
                                          recv_sizes.data(), recv_offsets.data());
                    exec->synchronize();
                }
        );
        mpi_comm.all_reduce(exec->get_master(), &elapsed_seconds, 1, MPI_MAX);
        state.SetIterationTime(elapsed_seconds);
    }
}

BENCHMARK_DEFINE_F(AllToAll, MPI_NeighborHood)(benchmark::State &state) {
    const auto num_kernels = state.range(1);

    for (auto _: state) {
        auto elapsed_seconds = timed(
                [&] {
                    for (int i = 0; i < num_kernels; ++i) {
                        send_buf.fill(mpi_comm.rank());
                        benchmark::DoNotOptimize(send_buf.get_data());
                    }

                    exec->synchronize();
                    {
                        auto g = exec->get_scoped_device_id_guard();
                        MPI_Neighbor_alltoallv(send_buf.get_data(), neighbor_send_sizes.data(),
                                               neighbor_send_offsets.data(),
                                               MPI_DOUBLE,
                                               recv_buf.get_data(), neighbor_recv_sizes.data(),
                                               neighbor_recv_offsets.data(),
                                               MPI_DOUBLE, neighbor_comm.get());
                    }
                    exec->synchronize();
                }
        );
        mpi_comm.all_reduce(exec->get_master(), &elapsed_seconds, 1, MPI_MAX);
        state.SetIterationTime(elapsed_seconds);
    }
}

BENCHMARK_DEFINE_F(AllToAll, MPI_Manual)(benchmark::State &state) {
    const auto num_kernels = state.range(1);

    for (auto _: state) {
        auto elapsed_seconds = timed(
                [&] {
                    for (int i = 0; i < num_kernels; ++i) {
                        send_buf.fill(mpi_comm.rank());
                        benchmark::DoNotOptimize(send_buf.get_data());
                    }

                    exec->synchronize();
                    {
                        auto g = exec->get_scoped_device_id_guard();
                        for (int i = 0; i < send_sizes.size(); ++i) {
                            if (send_sizes[i]) {
                                reqs.emplace_back(mpi_comm.i_send(exec, send_buf.get_data() + send_offsets[i],
                                                                  send_sizes[i], i, mpi_comm.rank()));
                            }
                            if (recv_sizes[i]) {
                                reqs.emplace_back(mpi_comm.i_recv(exec, recv_buf.get_data() + recv_offsets[i],
                                                                  recv_sizes[i], i, i));
                            }
                        }
                        gko::experimental::mpi::wait_all(reqs);
                    }
                    exec->synchronize();
                }
        );
        mpi_comm.all_reduce(exec->get_master(), &elapsed_seconds, 1, MPI_MAX);
        state.SetIterationTime(elapsed_seconds);
    }
}

BENCHMARK_DEFINE_F(AllToAll, NCCL)(benchmark::State &state) {
    const auto num_kernels = state.range(1);

    for (auto _: state) {
        auto elapsed_seconds = timed(
                [&] {
                    for (int i = 0; i < num_kernels; ++i) {
                        send_buf.fill(mpi_comm.rank());
                        benchmark::DoNotOptimize(send_buf.get_data());
                    }

                    ncclGroupStart();
                    for (int i = 0; i < send_sizes.size(); ++i) {
                        if (send_sizes[i]) {
                            ncclSend(send_buf.get_data() + send_offsets[i], send_sizes[i],
                                     ncclDouble, i, nccl_comm, exec->get_stream());
                        }
                        if (recv_sizes[i]) {
                            ncclRecv(recv_buf.get_data() + recv_offsets[i], recv_sizes[i],
                                     ncclDouble, i, nccl_comm, exec->get_stream());
                        }
                    }
                    ncclGroupEnd();
                    exec->synchronize();
                }
        );
        mpi_comm.all_reduce(exec->get_master(), &elapsed_seconds, 1, MPI_MAX);
        state.SetIterationTime(elapsed_seconds);
    }
}


// This reporter does nothing.
// We can use it to disable output from all but the root process
class NullReporter : public ::benchmark::BenchmarkReporter {
public:
    NullReporter() {}

    virtual bool ReportContext(const Context &) { return true; }

    virtual void ReportRuns(const std::vector<Run> &) {}

    virtual void Finalize() {}
};


int main(int argc, char **argv) {
    gko::experimental::mpi::environment env(argc, argv);

    gko::experimental::mpi::communicator mpi_comm(MPI_COMM_WORLD);
    ncclComm_t nccl_comm;

    auto exec = gko::__EXEC::create(
            gko::experimental::mpi::map_rank_to_device_id(mpi_comm.get(), gko::__EXEC::get_num_devices()),
            gko::ReferenceExecutor::create());
    auto g = exec->get_scoped_device_id_guard(); // seems to be required for any nccl calls

    ncclUniqueId id;
    if (mpi_comm.rank() == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm.get());
    ncclCommInitRank(&nccl_comm, mpi_comm.size(), id, mpi_comm.rank());

#define REGISTER_BENCHMARK(op, variant) \
benchmark::RegisterBenchmark(#op "/" #variant, [&](benchmark::State &st) { \
    op ## _ ## variant ## _Benchmark b;\
    b.exec = exec;\
    b.mpi_comm = mpi_comm;\
    b.nccl_comm = nccl_comm;\
    b.Run(st);\
})->UseManualTime()

    REGISTER_BENCHMARK(AllGather, None)->ArgsProduct({
                                                             {100, 1000, 10000},
                                                             {1,   5,    10}
                                                     });
    REGISTER_BENCHMARK(AllGather, MPI)->ArgsProduct({
                                                            {100, 1000, 10000},
                                                            {1,   5,    10}
                                                    });
    REGISTER_BENCHMARK(AllGather, NCCL)->ArgsProduct({
                                                             {100, 1000, 10000},
                                                             {1,   5,    10}
                                                     });

    REGISTER_BENCHMARK(AllReduce, MPI)->ArgsProduct({
                                                            {1, 10, 100},
                                                            {1, 5,  10}
                                                    });
    REGISTER_BENCHMARK(AllReduce, NCCL)->ArgsProduct({
                                                             {1, 10, 100},
                                                             {1, 5,  10}
                                                     });

    REGISTER_BENCHMARK(AllToAll, MPI)->ArgsProduct({
                                                           {10, 100, 1000},
                                                           {1,  5,   10}
                                                   });
    REGISTER_BENCHMARK(AllToAll, MPI_NeighborHood)->ArgsProduct({
                                                           {10, 100, 1000},
                                                           {1,  5,   10}
                                                   });
    REGISTER_BENCHMARK(AllToAll, MPI_Manual)->ArgsProduct({
                                                           {10, 100, 1000},
                                                           {1,  5,   10}
                                                   });
    REGISTER_BENCHMARK(AllToAll, NCCL)->ArgsProduct({
                                                            {10, 100, 1000},
                                                            {1,  5,   10}
                                                    });

    benchmark::Initialize(&argc, argv);
    if (mpi_comm.rank() == 0)
        // root process will use a reporter from the usual set provided by
        // ::benchmark
        ::benchmark::RunSpecifiedBenchmarks();
    else {
        // reporting from other processes is disabled by passing a custom reporter
        NullReporter null;
        ::benchmark::RunSpecifiedBenchmarks(&null);
    }
    benchmark::Shutdown();
    ncclCommDestroy(nccl_comm);
}
