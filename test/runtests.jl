using ThreadedSparseCSR
using SparseArrays, SparseMatricesCSR
using Test

@testset "matrix-vec multipy" begin
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
        y2 = copy(n)
        @test bmul!(y1, csrA, x, 0.5, 0.3) == mul!(y2, cscA, x, 0.5, 0.3)
        @test bmul!(y1, csrA, x, 0.5, 0.3) == mul!(y2, csrA, x, 0.5, 0.3)
    end

    @testset "3-arg, inplace mat-vec mul" begin
        y1 = rand(n)
        y2 = copy(n)
        @test bmul!(y1, csrA, x) == mul!(y2, cscA, x)
        @test bmul!(y1, csrA, x) == mul!(y2, csrA, x)
    end

    @testset "2-arg mat-vec mul" begin
        @test bmul(csrA, x) == cscA*x
        @test bmul(csrA, x) == csrA*x
    end

end
