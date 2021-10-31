function serial_csr_mv!(y::Vector, A::SparseMatrixCSR, x::Vector, alpha::Number, beta::Number)
    
    A.n == size(x, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())

    o = getoffset(A)
    
    @inbounds for row in 1:size(y, 1)

        accu = zero(eltype(y))
        for nz in nzrange(A, row)
            col = A.colval[nz] + o
            accu += A.nzval[nz]*x[col]
        end

        y[row] = alpha*accu + beta*y[row]
    
    end

    return y

end

function serial_csr_mv!(y::AbstractArray, A::SparseMatrixCSR, x::AbstractVector)
    
    serial_csr_mv!(y, A, x, true, false)

end

function serial_csr_mv(A::SparseMatrixCSR, x::AbstractVector)

    T = promote_type(eltype(A), eltype(x))
    m = A.m
    y = Vector{T}(undef, m)

    serial_csr_mv!(y, A, x, true, false)

end




function threaded_csr_mv!(y::AbstractArray, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
    
    A.n == size(x, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())

    o = getoffset(A)
    
    @Threads.threads for row in 1:size(y, 1)
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


function threaded_csr_mv!(y::AbstractArray, A::SparseMatrixCSR, x::AbstractVector)
    
    threaded_csr_mv!(y, A, x, true, false)

end

function threaded_csr_mv(A::SparseMatrixCSR, x::AbstractVector)

    T = promote_type(eltype(A), eltype(x))
    m = A.m
    y = Vector{T}(undef, m)

    threaded_csr_mv!(y, A, x, true, false)

end



function floops_csr_mv!(y::AbstractArray, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number; num_threads = Threads.nthreads())
    
    A.n == size(x, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())

    o = getoffset(A)
    
    @floop ThreadedEx(basesize = size(y, 1) ÷ num_threads) for row in 1:size(y, 1)
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


function floops_csr_mv!(y::AbstractArray, A::SparseMatrixCSR, x::AbstractVector; num_threads = Threads.nthreads())
    
    floops_csr_mv!(y, A, x, true, false; num_threads = num_threads)

end

function floops_csr_mv(A::SparseMatrixCSR, x::AbstractVector; num_threads = Threads.nthreads())

    T = promote_type(eltype(A), eltype(x))
    m = A.m
    y = Vector{T}(undef, m)

    floops_csr_mv!(y, A, x, true, false; num_threads = num_threads)

end

function batch_csr_mv!(y::AbstractArray, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number; num_threads = Threads.nthreads())
    
    A.n == size(x, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())

    o = getoffset(A)
    
    @batch minbatch = size(y, 1) ÷ num_threads for row in 1:size(y, 1)
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


function batch_csr_mv!(y::AbstractArray, A::SparseMatrixCSR, x::AbstractVector; num_threads = Threads.nthreads())
    
    batch_csr_mv!(y, A, x, true, false; num_threads = num_threads)

end

function batch_csr_mv(A::SparseMatrixCSR, x::AbstractVector; num_threads = Threads.nthreads())

    T = promote_type(eltype(A), eltype(x))
    m = A.m
    y = Vector{T}(undef, m)

    batch_csr_mv!(y, A, x, true, false; num_threads = num_threads)

end