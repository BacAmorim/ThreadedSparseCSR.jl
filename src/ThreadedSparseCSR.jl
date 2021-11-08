module ThreadedSparseCSR

using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using Polyester

export bmul!, bmul

include("matmul.jl")
include("multithread_mul.jl")


end
