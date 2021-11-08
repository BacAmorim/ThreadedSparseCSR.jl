import Base: *
import LinearAlgebra: mul!


abstract type ThreadingBackend end

struct BaseThreads <: ThreadingBackend end

function multithread_matmul(T::BaseThreads)

    @eval function  mul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
        return tmul!(y, A, x, alpha, beta)
    end

    @eval function  mul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector)
        return tmul!(y, A, x)
    end

    @eval function  *(A::SparseMatrixCSR, x::AbstractVector)
        return tmul(A, x)
    end

end


struct PolyesterThreads <: ThreadingBackend end

function multithread_matmul(T::PolyesterThreads)

    @eval function  mul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
        return bmul!(y, A, x, alpha, beta)
    end

    @eval function  mul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector)
        return bmul!(y, A, x)
    end

    @eval function  *(A::SparseMatrixCSR, x::AbstractVector)
        return bmul(A, x)
    end

end