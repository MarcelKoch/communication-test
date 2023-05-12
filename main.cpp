#include <ginkgo/ginkgo.hpp>
#include <rccl/rccl.h>

void incorrect_mpi_comm(gko::array<int>& send, gko::array<int>& recv, gko::experimental::mpi::communicator comm){
    auto exec = recv.get_executor();

    send.fill(comm.rank());

    comm.all_gather(exec, send.get_data(), 1, recv.get_data(), 1);
}

void correct_mpi_comm(gko::array<int>& send, gko::array<int>& recv, gko::experimental::mpi::communicator comm){
    auto exec = recv.get_executor();

    send.fill(comm.rank());

    exec->synchronize();  // necessary to ensure fill-kernel has completed before communication is started
    comm.all_gather(exec, send.get_data(), 1, recv.get_data(), 1);
}

void nccl_comm(gko::array<int>& send, gko::array<int>& recv, gko::experimental::mpi::communicator mpi_comm, ncclComm_t comm){
    auto exec = gko::as<gko::HipExecutor>(recv.get_executor());

    send.fill(mpi_comm.rank());

    ncclAllGather(send.get_data(), recv.get_data(), 1, ncclInt, comm, exec->get_stream());
}

int main(int argc, char** argv) {
    using namespace std::string_literals;
    gko::experimental::mpi::environment env{argc, argv};

    gko::experimental::mpi::communicator comm(MPI_COMM_WORLD);

    auto exec = gko::HipExecutor::create(
            gko::experimental::mpi::map_rank_to_device_id(comm.get(), 8), gko::ReferenceExecutor::create());
    auto g = exec->get_scoped_device_id_guard(); // seems to be required for any nccl calls

    ncclUniqueId id;
    if(comm.rank() == 0){
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm.get());

    ncclComm_t n_comm;
    ncclCommInitRank(&n_comm, comm.size(), id, comm.rank());

    gko::array<int> send(exec, comm.size());
    gko::array<int> recv(exec, comm.size());
    recv.fill(-1);
    exec->synchronize();

    if(argv[1] == "correct"s) {
        correct_mpi_comm(send, recv, comm);
    }
    if(argv[1] == "incorrect"s) {
        incorrect_mpi_comm(send, recv, comm);
    }
    if(argv[1] == "nccl"s) {
        nccl_comm(send, recv, comm, n_comm);
    }

    recv.set_executor(exec->get_master());
    for (int i = 0; i < comm.size(); ++i){
        if(recv.get_const_data()[i] != i){
            std::cerr << "Rank " << comm.rank() << ": " << recv.get_const_data()[i] << " (" << i << ")" << std::endl;
        }
    }

    ncclCommDestroy(n_comm);
}