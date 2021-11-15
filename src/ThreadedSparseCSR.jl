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

export tmul!, tmul
export bmul!, bmul
export BaseThreads, PolyesterThreads

include("threads_matmul.jl")
include("batch_matmul.jl")
include("multithread_matmul.jl")
include("set_num_threads.jl")
include("init_package.jl")

end
