# Multithreaded multiplication using Polyester.jl @batch
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


function csr_bmul!(y::AbstractArray, A::SparseMatrixCSR, x::AbstractVector)
    
    csr_bmul!(y, A, x, true, false)

end

function csr_bmul(A::SparseMatrixCSR, x::AbstractVector)

    T = promote_type(eltype(A), eltype(x))
    m = A.m
    y = Vector{T}(undef, m)

    csr_bmul!(y, A, x, true, false)

end