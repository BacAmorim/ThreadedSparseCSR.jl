module ThreadedSparseCSR

using SparseMatricesCSR
using Polyester
using LinearAlgebra

export csr_bmul!
#export csr_bmul

#include("matmul.jl")
#include("multithread_mul.jl")

using SparseMatricesCSR: nzrange

function csr_bmul!(y::AbstractArray, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
    
    A.n == size(x, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())

    o = getoffset(A)
    
    @batch for row in 1:size(y, 1)
        @inbounds begin
            accu = zero(eltype(y))
            for nz in nzrange(A, row)
                col = A.colval[nz] + o
                accu += A.nzval[nz]*x[col]
            end
            y[row] = alpha*accu + beta*y[row]
    
        end
    end

    return y

end

import LinearAlgebra: mul!

@eval begin

    function  mul!(y::AbstractArray, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
        return csr_bmul!(y, A, x, alpha, beta)
    end

    
end

end
