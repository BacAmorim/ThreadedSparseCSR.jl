# ThreadedSparseCSR

Provides a multithreaded version of sparse CSR matrix - vector multiplication in Julia. 

# Instalation
This package is not registered. To install it:
```
using Pkg
Pkg.add("https://github.com/BacAmorim/ThreadedSparseCSR.jl")
```

The CSR matrix format is implemented in the Julia package [SparseMatricesCSR](https://github.com/gridap/SparseMatricesCSR.jl), which must be installed for this package to work.

To enable multithreaded mat-vec multiplication, Julia must be initialized with threads, eitheir by setting the variable `JULIA_NUM_THREADS` or by inizializing julia as `julia -t n` (to start with `n` threads).

# Functionality
This packahe implements a multithreaded version of sparse matrix CSR - vector multiplication in Julia. 

The package exports the functions:
- `tmul!(y, A, x, [alpha], [beta])`, 5 argument (`y = alpha*A*x +beta*y `) and 3 argument (`y = A*x`) in-place multithreaded versions of `mul!`, using `Base.Threads` threading (using `@spawn`)
- `tmul(A, x)`, multithreaded version of `A*x`, using `Base.Threads` threading (using `@spawn`)
- `bmul!(y, A, x, [alpha], [beta])`, 5 argument (`y = alpha*A*x +beta*y `) and 3 argument (`y = A*x`) in-place multithreaded versions of `mul!`, using [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl) threading (using `@batch`)
- `bmul(A, x)`, multithreaded version of `A*x`, using [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl) threading (using `@batch`)

It is possible to overwrite the function `*` and `mul!` by their multithreaded versions. This is done using the function:
```
ThreadedSparseCSR.multithread_matmul(PolyesterThreads())
```
which overwrites `*` and `mul!` by `bmul` and `bmul!`, respectivelly;
```
ThreadedSparseCSR.multithread_matmul(BaseThreads())
```
which overwrites `*` and `mul!` by `tmul` and `tmul!`, respectivelly;
```
ThreadedSparseCSR.multithread_matmul()
```
by default, overwrites `*` and `mul!` by `bmul` and `bmul!`, respectivelly.

It is also possible to change the number of threads that are used, using the function
```
ThreadedSparseCSR.set_num_threads(4)
```
The number of threads that is being used is obtained via:
```
ThreadedSparseCSR.get_num_threads()
```

# Example Usage
```
using SparseArrays, SparseMatricesCSR, ThreadedSparseCSR
using BenchmarkTools

m, n = 1_000, 1_000
d = 0.01
num_nzs = floor(Int, m*n*d)
rows = rand(1:m, num_nzs)
cols = rand(1:n, num_nzs)
vals = rand(num_nzs)

cscA = sparse(rows, cols, vals, m, n)
csrA = sparsecsr(rows, cols, vals, m, n)
x = rand(n)

y1 = rand(n)
y2 = copy(y1)
y3 = copy(y1)


@btime mul!($y1, $csrA, $x, true, false) # non-threaded version
@btime bmul!($y2, $csrA, $x, true, false) # multithreaded version using Polyester.@batch
@btime tmul!($y3, $csrA, $x, true, false) # multithreaded version using Base.Threads.@spawn

ThreadedSparseCSR.multithread_matmul()
@btime mul!($y1, $csrA, $x, true, false) # multithreaded version using Polyester.@batch

ThreadedSparseCSR.multithread_matmul(BaseThreads())
@btime mul!($y1, $csrA, $x, true, false) # multithreaded version using Base.Threads.@spawn

ThreadedSparseCSR.multithread_matmul(PolyesterThreads())
@btime mul!($y1, $csrA, $x, true, false) # multithreaded version using Polyester.@batch


# Change the number of threads:
ThreadedSparseCSR.get_num_threads()
ThreadedSparseCSR.set_num_threads(4)
@btime mul!($y1, $csrA, $x, true, false) # multithreaded version using Polyester.@batch

```

# Benchmarks

Let us compare the performance of multithreaded sparse CSR matrix - vec as implemented in this package, with the non-threaded version and the multithreaded sparse CSC matrix - vec multiplication provided by [MKLSparse.jl](https://github.com/gridap/SparseMatricesCSR.jl) (both for non-transposed and transposed matrix). 

![benchmark_csr_matvec.png](https://github.com/BacAmorim/ThreadedSparseCSR.jl/blob/main/benchmark_csr_matvec.png?raw=true)

Code for benchmark:
```
using LinearAlgebra, SparseArrays, SparseMatricesCSR, ThreadedSparseCSR
using MKLSparse # to enable multithreaded Sparse CSC Matrix-Vec multiplication
using BenchmarkTools, PyPlot

function benchmark_csr_mv(sizes, densities)
    
    times_csc = zeros(length(sizes), length(densities))
    times_csc_transpose = zeros(length(sizes), length(densities))
    times_csr_mul = zeros(length(sizes), length(densities))
    times_csr_bmul = zeros(length(sizes), length(densities))
    times_csr_tmul = zeros(length(sizes), length(densities))
    
    for (j, d) in enumerate(densities)
        for (i, n) in enumerate(sizes)
            num_nzs = floor(Int, n*n*d)
            rows = rand(1:n, num_nzs)
            cols = rand(1:n, num_nzs)
            vals = rand(num_nzs)
            
            cscA = sparse(rows, cols, vals, n, n)
            cscAt = transpose(cscA)
            csrA = sparsecsr(rows, cols, vals, n, n)
            
            x = rand(n)
            y1 = zeros(n)
            y2 = zeros(n)
            y3 = zeros(n)
            y4 = zeros(n)
            y5 = zeros(n)
            
            b_csc = @benchmark mul!($y1, $cscA, $x, true, false)
            times_csc[i, j] = minimum(b_csc).time/1000 # time in microseconds
            
            b_csc_transpose = @benchmark mul!($y2, $cscAt, $x, true, false)
            times_csc_transpose[i, j] = minimum(b_csc_transpose).time/1000 # time in microseconds
            
            b_csr_mul = @benchmark mul!($y3, $csrA, $x, true, false)
            times_csr_mul[i, j] = minimum(b_csr_mul).time/1000 # time in microseconds
            
            b_csr_bmul = @benchmark bmul!($y4, $csrA, $x, true, false)
            times_csr_bmul[i, j] = minimum(b_csr_bmul).time/1000 # time in microseconds
            
            b_csr_tmul = @benchmark tmul!($y5, $csrA, $x, true, false)
            times_csr_tmul[i, j] = minimum(b_csr_tmul).time/1000 # time in microseconds
            
        end
    end
    
    return times_csc, times_csc_transpose, times_csr_mul, times_csr_bmul, times_csr_tmul
    
end

sizes = [1_000, 5_000, 10_000, 50_000, 100_000]
densities = [0.01, 0.001, 0.0001]

times_csc, times_csc_transpose, times_csr_mul, times_csr_bmul, times_csr_tmul = benchmark_csr_mv(sizes, densities)

f, ax = subplots(1, 3, figsize = (13, 5))

for (i, d) in enumerate(densities)
    ax[i].loglog(sizes, times_csc[:, i], marker = "v", label = "MKLSparse, CSC")
    ax[i].loglog(sizes, times_csc_transpose[:, i], marker = "^", label = "MKLSparse, transpose(CSC)")
    ax[i].loglog(sizes, times_csr_mul[:, i], marker = "h", label = "non-threaded mul (CSR)")
    ax[i].loglog(sizes, times_csr_bmul[:, i], marker = "s", label = "bmul (CSR)")
    ax[i].loglog(sizes, times_csr_tmul[:, i], marker = "o", label = "tmul (CSR)")
    
    ax[i].set_title("non-zero density = $(d)")
    ax[i].set_xlabel("matrix size")
    ax[i].set_ylabel("minimum time [μs]")
    ax[i].set_xticks(sizes)
    ax[i].set_xticklabels(sizes)
end

legend()
tight_layout()
savefig("benchmark_csr_matvec.png", dpi = 300)
```

# Acknowlegments
This package was influenced and inspired by:
- [ThreadedSparseArrays.jl](https://github.com/jagot/ThreadedSparseArrays.jl)
- [SparseMatricesCSR.jl](https://github.com/gridap/SparseMatricesCSR.jl)
- [MKLSparse.jl](https://github.com/gridap/SparseMatricesCSR.jl)



