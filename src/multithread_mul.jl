import Base: *
import LinearAlgebra: mul!

function multithread_matmul()

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