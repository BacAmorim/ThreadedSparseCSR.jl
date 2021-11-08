# ThreadedSparseCSR

Provides a multithreaded version of sparse CSR matrix vector multiplication in Julia. 

The CSR matrix format is implemented in the Julia package https://github.com/gridap/SparseMatricesCSR.jl

This package exports the functions:
- `tmul!(y, A, x, [alpha], [beta])`, 5 argument (`y = alpha*A*x +beta*y `) and 3 argument (`y = A*x`) in-place multithreaded versions of `mul!`, using `Base.Threads` threading (using `@spawn`)
- `tmul(A, x)`, multithreaded versions of `A*x`, using `Base.Threads` threading (using `@spawn`)
- `bmul!(y, A, x, [alpha], [beta])`, 5 argument (`y = alpha*A*x +beta*y `) and 3 argument (`y = A*x`) in-place multithreaded versions of `mul!`, using https://github.com/JuliaSIMD/Polyester.jl threading (using `@batch`)
- `bmul(A, x)`, multithreaded versions of `A*x`, using https://github.com/JuliaSIMD/Polyester.jl threading (using `@batch`)

It is also possible to overwrite the function `*` and `mul!` by their multithreaded versions. This is done using the function:
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
by default, overwrites `*` and `mul!` by `tmul` and `tmul`, respectivelly.


