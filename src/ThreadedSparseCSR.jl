module ThreadedSparseCSR

using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using Polyester

using SparseMatricesCSR: nzrange
import Base: *
import LinearAlgebra: mul! 

abstract type ThreadingBackend end
struct BaseThreads <: ThreadingBackend end
struct PolyesterThreads <: ThreadingBackend end

DefaultThreadingBackend() = PolyesterThreads()

export BaseThreads, tmul!, tmul
export PolyesterThreads, bmul!, bmul

include("threads_matmul.jl")
include("batch_matmul.jl")
include("set_num_threads.jl")

end
