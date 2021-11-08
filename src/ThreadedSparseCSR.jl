module ThreadedSparseCSR

using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using Polyester

using SparseMatricesCSR: nzrange
import Base: *
import LinearAlgebra: mul! 


include("threads_matmul.jl")
export tmul!, tmul

include("batch_matmul.jl")
export bmul!, bmul

include("multithread_matmul.jl")
export BaseThreads, PolyesterThreads

include("set_num_threads.jl")

end
