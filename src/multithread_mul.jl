import Base: *
import SparseMatricesCSR: mul!

function multithread_mul!()

    @eval begin

        function  mul!(y::AbstractArray, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
            return csr_bmul!(y, A, x, alpha, beta)
        end

        function mul!(y::AbstractArray, A::SparseMatrixCSR, x::AbstractVector)
            return csr_bmul!(y, A, x)
        end

        function (*)(A::SparseMatrixCSR, x::AbstractVector)
            return csr_bmul(A, x)
        end
    end

    println("CSR mat-vec multiplication, mul! and *,  is now multithreaded!")

    return nothing

end

@eval begin

    function  mul!(y::AbstractArray, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
        return csr_bmul!(y, A, x, alpha, beta)
    end

    function mul!(y::AbstractArray, A::SparseMatrixCSR, x::AbstractVector)
        return csr_bmul!(y, A, x)
    end

    function (*)(A::SparseMatrixCSR, x::AbstractVector)
        return csr_bmul(A, x)
    end
    
end