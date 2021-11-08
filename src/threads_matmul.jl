# Multithreaded multiplication using Threads.@spawn (based on https://github.com/jagot/ThreadedSparseArrays.jl)

## * Threading utilities
struct RangeIterator
    k::Int
    d::Int
    r::Int
end

"""
    RangeIterator(n::Int,k::Int)
Returns an iterator splitting the range `1:n` into `min(k,n)` parts of (almost) equal size.
"""
RangeIterator(n::Int, k::Int) = RangeIterator(min(n,k),divrem(n,k)...)
Base.length(it::RangeIterator) = it.k
endpos(it::RangeIterator, i::Int) = i*it.d+min(i,it.r)
Base.iterate(it::RangeIterator, i::Int=1) = i>it.k ? nothing : (endpos(it,i-1)+1:endpos(it,i), i+1)

function tmul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
    
    A.n == size(x, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())

    o = getoffset(A)

    @sync for r in RangeIterator(size(y, 1), matmul_num_threads[])
        Threads.@spawn for row in r
            @inbounds begin
                accu = zero(eltype(y))
                for nz in nzrange(A, row)
                    col = A.colval[nz] + o
                    accu += A.nzval[nz]*x[col]
                end
                y[row] = alpha*accu + beta*y[row]
            end
        end
    end

    return y

end

# sparse mat-vec multiplication
function tmul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector)
    
    tmul!(y, A, x, true, false)

end

function tmul(A::SparseMatrixCSR, x::AbstractVector)

    y = similar(x, promote_type(eltype(A), eltype(x)), size(A, 1))

    tmul!(y, A, x, true, false)

end



# function to overwrite * and mul!
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


function  multithread_matmul()

    multithread_matmul(DefaultThreadingBackend())

end