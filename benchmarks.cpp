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

        send_sizes[target_rank(mod(r_x + 1, offset), r_y)] = size;
        send_sizes[target_rank(mod(r_x - 1, offset), r_y)] = size;
        send_sizes[target_rank(r_x, mod(r_y + 1, offset))] = size;
        send_sizes[target_rank(r_x, mod(r_y - 1, offset))] = size;

        recv_sizes[target_rank(mod(r_x + 1, offset), r_y)] = size;
        recv_sizes[target_rank(mod(r_x - 1, offset), r_y)] = size;
        recv_sizes[target_rank(r_x, mod(r_y + 1, offset))] = size;
        recv_sizes[target_rank(r_x, mod(r_y - 1, offset))] = size;

        std::partial_sum(send_sizes.begin(), send_sizes.end(), send_offsets.begin() + 1);
        std::partial_sum(recv_sizes.begin(), recv_sizes.end(), recv_offsets.begin() + 1);

        send_buf = gko::array<double>(exec, send_offsets.back());
        recv_buf = gko::array<double>(exec, recv_offsets.back());
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
                    for (int i = 0; i < send_sizes.size(); ++i){
                        if (send_sizes[i]){
                            ncclSend(send_buf.get_data() + send_offsets[i], send_sizes[i],
                                     ncclDouble, i, nccl_comm, exec->get_stream());
                        }
                        if (recv_sizes[i]){
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
            gko::experimental::mpi::map_rank_to_device_id(mpi_comm.get(), 8),
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
                                                           {1, 5, 10}
                                                   });
    REGISTER_BENCHMARK(AllToAll, NCCL)->ArgsProduct({
                                                           {10, 100, 1000},
                                                           {1, 5, 10}
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
