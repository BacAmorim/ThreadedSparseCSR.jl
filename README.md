# ThreadedSparseCSR

Provides a multithreaded version of sparse CSR matrix vector multiplication in Julia. 

The CSR matrix format is implemented in the Julia package https://github.com/gridap/SparseMatricesCSR.jl

This package exports the functions:
- `tmul!(y, A, x, [alpha], [beta])`, 5 argument (`y = alpha*A*x +beta*y `) and 3 argument (`y = A*x`) in-place multithreaded versions of `mul!`, using `Base.Threads` threading (using `@spawn`)
- `tmul(A, x)`, multithreaded versions of `A*x`, using `Base.Threads` threading (using `@spawn`)
- `bmul!(y, A, x, [alpha], [beta])`, 5 argument (`y = alpha*A*x +beta*y `) and 3 argument (`y = A*x`) in-place multithreaded versions of `mul!`, using https://github.com/JuliaSIMD/Polyester.jl threading (using `@batch`)
- `bmul(A, x)`, multithreaded versions of `A*x`, using https://github.com/JuliaSIMD/Polyester.jl threading (using `@batch`)