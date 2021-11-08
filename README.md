# ThreadedSparseCSR

Provides a multithreaded version of sparse CSR matrix vector multiplication in Julia. 

The CSR matrix format is implemented in the Julia package https://github.com/gridap/SparseMatricesCSR.jl

This package exports the functions:
- `tmul!(y, A, x, [alpha], [beta])`, 5 argument (`y = alpha*A*x +beta*y `) and 3 argument (`y = A*x`) in-place multithreaded versions of `mul!`, using `Base.Threads` threading (using `@spawn`)
- `tmul(A, x)`, multithreaded versions of `A*x`, using `Base.Threads` threading (using `@spawn`)
- `bmul!(y, A, x, [alpha], [beta])`, 5 argument (`y = alpha*A*x +beta*y `) and 3 argument (`y = A*x`) in-place multithreaded versions of `mul!`, using https://github.com/JuliaSIMD/Polyester.jl threading (using `@batch`)
- `bmul(A, x)`, multithreaded versions of `A*x`, using https://github.com/JuliaSIMD/Polyester.jl threading (using `@batch`)

It is possible to overwrite the function `*` and `mul!` by their multithreaded versions. This is done using the function:
```
ThreadedSparseCSR.multithread_matmul(PolyesterThreads())
```
which overwrites `*` and `mul!` by `bmul` and `bmul`, respectivelly;
```
ThreadedSparseCSR.multithread_matmul(BaseThreads())
```
which overwrites `*` and `mul!` by `tmul` and `tmul`, respectivelly;
```
ThreadedSparseCSR.multithread_matmul()
```
by default, overwrites `*` and `mul!` by `bmul` and `bmul`, respectivelly.

It is also possible to change the number of threads that are used, using the function
```
ThreadedSparseCSR.set_num_threads(4)
```
The number of threads used is obtained via:
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

