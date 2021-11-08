module ThreadedSparseCSR

using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using Polyester

export bmul!, bmul

include("batch_matmul.jl")
include("multithread_mul.jl")
include("set_num_threads.jl")


end
