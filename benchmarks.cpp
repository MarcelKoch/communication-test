#include <ginkgo/ginkgo.hpp>
#include __NCCL_INC
#include <benchmark/benchmark.h>

class Fixture : public benchmark::Fixture {
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


BENCHMARK_DEFINE_F(Fixture, None)(benchmark::State &state) {
    const auto num_kernels = state.range(1);

    for (auto _: state) {
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < num_kernels; ++i) {
            send_buf.fill(mpi_comm.rank());
        }
        for (int i = 0; i < num_kernels; ++i) {
            recv_buf.fill(mpi_comm.rank());
        }
        exec->synchronize();
        mpi_comm.synchronize();
        auto end = std::chrono::steady_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<
                std::chrono::duration<double>>(end - start).count();
        mpi_comm.all_reduce(exec->get_master(), &elapsed_seconds, 1, MPI_MAX);
        state.SetIterationTime(elapsed_seconds);
    }
}

BENCHMARK_DEFINE_F(Fixture, ExtraSync)(benchmark::State &state) {
    const auto num_kernels = state.range(1);

    for (auto _: state) {
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < num_kernels; ++i) {
            send_buf.fill(mpi_comm.rank());
        }
        exec->synchronize();
        for (int i = 0; i < num_kernels; ++i) {
            recv_buf.fill(mpi_comm.rank());
        }
        exec->synchronize();
        mpi_comm.synchronize();
        auto end = std::chrono::steady_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<
                std::chrono::duration<double>>(end - start).count();
        mpi_comm.all_reduce(exec->get_master(), &elapsed_seconds, 1, MPI_MAX);
        state.SetIterationTime(elapsed_seconds);
    }
}

BENCHMARK_DEFINE_F(Fixture, MPI)(benchmark::State &state) {
    const auto num_kernels = state.range(1);

    for (auto _: state) {
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < num_kernels; ++i) {
            send_buf.fill(mpi_comm.rank());
        }

        exec->synchronize();
        mpi_comm.all_gather(exec, send_buf.get_data(), send_buf.get_num_elems(), recv_buf.get_data(),
                            send_buf.get_num_elems());

        for (int i = 0; i < num_kernels; ++i) {
            recv_buf.fill(mpi_comm.rank());
        }

        exec->synchronize();
        mpi_comm.synchronize();
        auto end = std::chrono::steady_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<
                std::chrono::duration<double>>(end - start).count();
        mpi_comm.all_reduce(exec->get_master(), &elapsed_seconds, 1, MPI_MAX);
        state.SetIterationTime(elapsed_seconds);
    }
}

BENCHMARK_DEFINE_F(Fixture, NCCL)(benchmark::State &state) {
    const auto num_kernels = state.range(1);

    for (auto _: state) {
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < num_kernels; ++i) {
            send_buf.fill(mpi_comm.rank());
        }

        auto g = exec->get_scoped_device_id_guard();
        ncclAllGather(send_buf.get_data(), recv_buf.get_data(), send_buf.get_num_elems(), ncclDouble, nccl_comm,
                      exec->get_stream());

        for (int i = 0; i < num_kernels; ++i) {
            recv_buf.fill(mpi_comm.rank());
        }

        exec->synchronize();
        mpi_comm.synchronize();
        auto end = std::chrono::steady_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<
                std::chrono::duration<double>>(end - start).count();
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

    benchmark::RegisterBenchmark("none", [&](benchmark::State &st) {
        Fixture_None_Benchmark b;
        b.exec = exec;
        b.mpi_comm = mpi_comm;
        b.nccl_comm = nccl_comm;
        b.Run(st);
    })->ArgsProduct({
                            {100, 1000, 10000},
                            {5,   10,    20}
                    })->UseManualTime();
    benchmark::RegisterBenchmark("extra-synch", [&](benchmark::State &st) {
        Fixture_ExtraSync_Benchmark b;
        b.exec = exec;
        b.mpi_comm = mpi_comm;
        b.nccl_comm = nccl_comm;
        b.Run(st);
    })->ArgsProduct({
                            {100, 1000, 10000},
                            {5,   10,    20}
                    })->UseManualTime();
    benchmark::RegisterBenchmark("mpi", [&](benchmark::State &st) {
        Fixture_MPI_Benchmark b;
        b.exec = exec;
        b.mpi_comm = mpi_comm;
        b.nccl_comm = nccl_comm;
        b.Run(st);
    })->ArgsProduct({
                            {100, 1000, 10000},
                            {5,   10,    20}
                    })->UseManualTime();
    benchmark::RegisterBenchmark("nccl", [&](benchmark::State &st) {
        Fixture_NCCL_Benchmark b;
        b.exec = exec;
        b.mpi_comm = mpi_comm;
        b.nccl_comm = nccl_comm;
        b.Run(st);
    })->ArgsProduct({
                            {100, 1000, 10000},
                            {5,   10,    20}
                    })->UseManualTime();

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
