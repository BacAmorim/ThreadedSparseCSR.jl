# Multithreaded multiplication using Polyester.jl @batch
using SparseMatricesCSR: nzrange

# sparse mat-vec multiplication
function bmul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
    
    A.n == size(x, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())

    o = getoffset(A)
    
    @batch minbatch = size(y, 1) ÷ matmul_num_threads[] for row in 1:size(y, 1)
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

    y = similar(x, promote_type(eltype(A), eltype(x)), size(A, 1))
    
    return bmul!(y, A, x, true, false)

end