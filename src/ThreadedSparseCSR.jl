module ThreadedSparseCSR

using SparseMatricesCSR
using FLoops
using Polyester

import SparseMatricesCSR: nzrange

#export get_num_threads, set_num_threads
export serial_csr_mv, threaded_csr_mv, floops_csr_mv, batch_csr_mv

include("matmul.jl")

end
