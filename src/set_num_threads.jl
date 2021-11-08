matmul_num_threads = Ref(Threads.nthreads())

function set_num_threads(n::Int)

    0 < n <=  Threads.nthreads() || throw(DomainError("The numbers of threads must be > 0 and <= $(Threads.nthreads())."))

    global matmul_num_threads[] = n

end


function get_num_threads()

    return matmul_num_threads[]
    
end