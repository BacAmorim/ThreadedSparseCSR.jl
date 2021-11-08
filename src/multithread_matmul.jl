abstract type ThreadingBackend end
struct BaseThreads <: ThreadingBackend end
struct PolyesterThreads <: ThreadingBackend end

DefaultThreadingBackend() = PolyesterThreads()



# function to overwrite * and mul!

## with Base.threads
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

## with Polyester
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

## with default threading backend
function  multithread_matmul()

    multithread_matmul(DefaultThreadingBackend())

end