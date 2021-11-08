using ThreadedSparseCSR
using SparseArrays, SparseMatricesCSR
using Test

@testset "Base.Threads matrix-vec multipy" begin
    m, n = 1_000, 1_000
    d = 0.01
    num_nzs = floor(Int, m*n*d)
    rows = rand(1:m, num_nzs)
    cols = rand(1:n, num_nzs)
    vals = rand(num_nzs)

    cscA = sparse(rows, cols, vals, m, n)
    csrA = sparsecsr(rows, cols, vals, m, n)
    x = rand(n)

    @testset "5-arg, inplace mat-vec mul" begin
        y1 = rand(n)
        y2 = copy(y1)
        @test tmul!(y1, csrA, x, 0.5, 0.3) ≈ SparseArrays.mul!(y2, cscA, x, 0.5, 0.3)

        y1 = rand(n)
        y2 = copy(y1)
        @test tmul!(y1, csrA, x, 0.5, 0.3) ≈ SparseMatricesCSR.mul!(y2, csrA, x, 0.5, 0.3)
    end

    @testset "3-arg, inplace mat-vec mul" begin
        y1 = rand(n)
        y2 = copy(y1)
        @test tmul!(y1, csrA, x) ≈ SparseArrays.mul!(y2, cscA, x)
        
        y1 = rand(n)
        y2 = copy(y1)
        @test tmul!(y1, csrA, x) ≈ SparseMatricesCSR.mul!(y2, csrA, x)
    end

   @testset "2-arg mat-vec mul" begin
        @test tmul(csrA, x) ≈ cscA*x
        @test tmul(csrA, x) ≈ SparseMatricesCSR.:*(csrA, x)
    end


end



@testset "Polyester matrix-vec multipy" begin
    m, n = 1_000, 1_000
    d = 0.01
    num_nzs = floor(Int, m*n*d)
    rows = rand(1:m, num_nzs)
    cols = rand(1:n, num_nzs)
    vals = rand(num_nzs)

    cscA = sparse(rows, cols, vals, m, n)
    csrA = sparsecsr(rows, cols, vals, m, n)
    x = rand(n)

    @testset "5-arg, inplace mat-vec mul" begin
        y1 = rand(n)
        y2 = copy(y1)
        @test bmul!(y1, csrA, x, 0.5, 0.3) ≈ SparseArrays.mul!(y2, cscA, x, 0.5, 0.3)

        y1 = rand(n)
        y2 = copy(y1)
        @test bmul!(y1, csrA, x, 0.5, 0.3) ≈ SparseMatricesCSR.mul!(y2, csrA, x, 0.5, 0.3)
    end

    @testset "3-arg, inplace mat-vec mul" begin
        y1 = rand(n)
        y2 = copy(y1)
        @test bmul!(y1, csrA, x) ≈ SparseArrays.mul!(y2, cscA, x)
        
        y1 = rand(n)
        y2 = copy(y1)
        @test bmul!(y1, csrA, x) ≈ SparseMatricesCSR.mul!(y2, csrA, x)
    end

   @testset "2-arg mat-vec mul" begin
        @test bmul(csrA, x) ≈ cscA*x
        @test bmul(csrA, x) ≈ SparseMatricesCSR.:*(csrA, x)
    end


end
