module ThreadedSparseCSR

using SparseArrays
using SparseMatricesCSR
using Polyester
using LinearAlgebra

export bmul!, bmul
#export csr_bmul

#include("matmul.jl")
#include("multithread_mul.jl")

using SparseMatricesCSR: nzrange

function bmul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
    
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

function bmul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector)
    
    return bmul!(y, A, x, true, false)

end

function bmul(A::SparseMatrixCSR, x::AbstractVector)

    T = promote_type(eltype(A), eltype(x))
    y = similar(x, T)
    
    return bmul!(y, A, x, true, false)

end

import LinearAlgebra: mul!

function multithread_matmul()

    @eval function  mul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
        return bmul!(y, A, x, alpha, beta)
    end

end

end
