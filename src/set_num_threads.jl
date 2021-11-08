matmul_num_threads = Ref(Threads.nthreads())

"""
    ThreadedSparseCSR.set_num_threads(n::Int)

Sets the number of threads used in sparse csrt matrix - vector multiplication.
"""
function set_num_threads(n::Int)

    0 < n <=  Threads.nthreads() || throw(DomainError("The numbers of threads must be > 0 and <= $(Threads.nthreads())."))

    global matmul_num_threads[] = n

end

"""
    ThreadedSparseCSR.get_num_threads()

Gets the number of threads used in sparse csrt matrix - vector multiplication.
"""
function get_num_threads()

    return matmul_num_threads[]
    
end