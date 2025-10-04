export ica_ken, generate_data

# # Optional: Uncomment the necessary GPU library
# # using CUDA  # For NVIDIA GPUs
# # using Metal # For macOS GPUs

# function create_array(data::Union{AbstractMatrix{T}, AbstractVector{T}}, use_gpu::Bool=false, use_metal::Bool=false) where T <: AbstractFloat
#     if use_gpu
#         @assert isdefined(Main, :CUDA) "CUDA.jl is not loaded. Please install and load CUDA.jl."
#         return isa(data, CuArray) ? data : CuArray(data)
#     elseif use_metal
#         @assert isdefined(Main, :Metal) "Metal.jl is not loaded. Please install and load Metal.jl."
#         return isa(data, MtlArray) ? data : MtlArray(data)
#     else
#         return isa(data, Array) ? data : Array(data)
#     end
# end

# # Define the icastruct
# struct icastruct{T <: AbstractFloat, A <: AbstractArray{T}}
#     X :: A
#     W :: A
#     Y :: A
#     D :: A
#     D_inv :: A
#     D_inv_sqrt :: A
#     M_storage1 :: A
#     M_storage2 :: A
#     M_storage3 :: A
#     M_storage4 :: A
#     M_storage5 :: A
#     Xt_storage :: A
#     E_storage :: AbstractVector{T}
#     G :: A
#     G_old :: A
#     psiY :: A
#     psidY :: A
#     direction :: A
#     I_storage :: A
#     h :: A
# end

# function icastruct(X::AbstractMatrix{T}, use_gpu::Bool=false, use_metal::Bool=false) where T <: AbstractFloat
#     # Convert input matrix to the appropriate device (CPU/GPU)
#     X_device = create_array(X, use_gpu, use_metal)
#     m, n = size(X_device)
#     @assert n ≥ m "Expect more samples (columns) than dimensions (rows) in X"
    
#     if use_gpu || use_metal
#         # perform in cpu
#         X_cpu = Array(X_device)
#         D_cpu = T(0.5) / n * (X_cpu * transpose(X_cpu))
#         D_inv_cpu = inv(D_cpu)
#         D_inv_sqrt_cpu = sqrt(D_inv_cpu)
#         # back Metal
#         D = create_array(D_cpu, use_gpu, use_metal)
#         D_inv = create_array(D_inv_cpu, use_gpu, use_metal)
#         D_inv_sqrt = create_array(D_inv_sqrt_cpu, use_gpu, use_metal)
#     else
#         D = T(0.5) / n * (X * transpose(X))
#         D_inv = inv(D)
#         D_inv_sqrt = sqrt(D_inv)
#     end
    
#     # Initialize other fields with consistent types
#     W = create_array(Matrix{T}(I, m, m), use_gpu, use_metal)
#     M_storage1 = create_array(zeros(T, m, m), use_gpu, use_metal)
#     M_storage2 = create_array(zeros(T, m, m), use_gpu, use_metal)
#     M_storage3 = create_array(zeros(T, m, m), use_gpu, use_metal)
#     M_storage4 = create_array(zeros(T, m, n), use_gpu, use_metal)
#     M_storage5 = create_array(zeros(T, m, n), use_gpu, use_metal)
#     Xt_storage = create_array(zeros(T, n, m), use_gpu, use_metal)
#     I_storage = create_array(Matrix{T}(I, m, m), use_gpu, use_metal)
#     E_storage = create_array(zeros(T, m), use_gpu, use_metal)  # Reshape as 2D array
    
#     Y = similar(X_device)
#     copyto!(Y, X_device)

#     # Return the icastruct instance
#     return icastruct(X_device, 
#         W, 
#         Y,
#         D, 
#         D_inv, 
#         D_inv_sqrt, 
#         M_storage1, 
#         M_storage2, 
#         M_storage3, 
#         M_storage4, 
#         M_storage5, 
#         Xt_storage, 
#         E_storage,
#         similar(W),
#         similar(W),
#         similar(X),
#         similar(X),
#         similar(W),
#         I_storage,
#         similar(W))
# end

# eltype(::icastruct{T, A}) where {T, A} = T

# # Helper function to compute the L-BFGS direction
# function _l_bfgs_direction(icas::icastruct{T, M}, s_list, y_list, r_list, precon, lambda_min)  where {T <: AbstractFloat, M <: AbstractMatrix{T}}
#     q = copy(icas.G)
#     a_list = Float64[]
#     for (s, y, r) in zip(reverse(s_list), reverse(y_list), reverse(r_list))
#         alpha = r * sum(s .* q)
#         push!(a_list, alpha)
#         q .-= alpha .* y
#     end
#     z = solveh(q, icas.h)
    
#     for (s, y, r, alpha) in zip(s_list, y_list, r_list, reverse(a_list))
#         beta = r * sum(y .* z)
#         z .+= (alpha - beta) .* s
#     end
#     return z
# end

# function score!(icas::icastruct{T, M}) where {T <: AbstractFloat, M <: AbstractMatrix{T}}
#     if isa(icas.Y, CuArray) || occursin("Mtl", string(typeof(icas.Y)))
#         icas.psiY .= tanh.(icas.Y ./ T(2))
#     else
#         @. icas.M_storage4 .= icas.Y ./ T(2)
#         AppleAccelerate.tanh!(icas.psiY, icas.M_storage4)
#     end
# end

# function score_der!(Y::AbstractMatrix{T}, X::AbstractMatrix{T}) where {T <: AbstractFloat}
#     if isa(X, CuArray) || occursin("Mtl", string(typeof(X)))
#         Y .= (one(T) .- X.^T(2)) ./ T(2)
#     else
#         @turbo for j in axes(X, 2), i in axes(X, 1)
#             Y[i, j] = (one(T) - X[i, j]^T(2)) / T(2)
#         end
#     end
# end

# function loss(Y::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T <: AbstractFloat}
#     n = size(Y, 2)

#     if isa(W, MtlMatrix)  
#         W_cpu = Array(W)  
#         Y_cpu = Array(Y) 
#         log_det, _ = logabsdet(W_cpu)
#         logcoshY = abs.(Y_cpu) .+ T(2) .* log1p.(exp.(-abs.(Y_cpu)))
#         total = sum(logcoshY) / n
#     else  
#         log_det, _ = logabsdet(W)
#         logcoshY = abs.(Y) .+ T(2) .* log1p.(exp.(-abs.(Y)))
#         total = sum(logcoshY) / n
#     end

#     return log_det - total
# end

# # Gradient computation
# function gradient(icas::icastruct{T, M}) where {T <: AbstractFloat, M <: AbstractMatrix{T}}
#     m, n = size(icas.Y)
    
#     copyto!(icas.M_storage1, icas.I_storage) #   icas.M_storage1 .= I(m)
#     alpha = -one(T) / T(n)
#     beta = one(T)
#     mul!(icas.M_storage1, icas.psiY, transpose(icas.Y), alpha, beta)
    
#     return icas.M_storage1
# end

# function compute_eigenvalues!(eigenvalues, h::AbstractMatrix{T}) where T <: AbstractFloat
    
#     if isa(h, MtlMatrix)
#         eigenvalues .= T(0.5) .* (h .+ h' .- sqrt.((h .- h').^T(2) .+ T(4.0)))
#     else
#         n, m = size(h)
#         @inbounds for i in 1:n
#             @simd for j in 1:m
#                 a = h[i, j]
#                 b = h[j, i]
#                 diff = a - b
#                 sum_ = a + b
#                 sq = diff^T(2) + T(4.0)
#                 eigenvalues[i, j] = T(0.5) * (sum_ - sqrt(sq))
#             end
#         end
#     end
#     return eigenvalues
# end

# # Compute Hessian approximation
# function compute_h(icas::icastruct{T, M}, precon::Int)  where {T <: AbstractFloat, M <: AbstractMatrix{T}}
#     m, n = size(icas.Y)
#     if precon == 2
#         @. icas.M_storage4 .= icas.Y ^ T(2)
#         mul!(icas.M_storage1, icas.psidY, transpose(icas.M_storage4), T(1)/n, T(0))
#         return icas.M_storage1
#     else
#         @. icas.M_storage4 .= icas.Y .^ T(2)
#         sigma2 = mean(icas.M_storage4, dims=2)
#         psidY_mean = mean(icas.psidY, dims=2)
#         @. icas.h .= psidY_mean .* sigma2'
#         diagonal_term = mean(icas.M_storage4)
#         for i in 1:m
#             icas.h[i, i] .= diagonal_term
#         end
#         return icas.h
#     end
# end

# function regularize_h(icas::icastruct{T, M}, lambda_min::T, mode::Int=0)  where {T <: AbstractFloat, M <: AbstractMatrix{T}}
#     """
#     Regularizes the Hessian approximation `h` using the constant `lambda_min`.
#     Mode selects the regularization algorithm:
#     0 -> Shift each eigenvalue below `lambda_min` to `lambda_min`.
#     1 -> Add `lambda_min * I` to `h`.
#     """
#     if mode == 0
        
#         # Compute the eigenvalues of the Hessian
#         eigenvalues = similar(icas.h)
#         compute_eigenvalues!(eigenvalues, icas.h)
        
#         # Regularize
#         problematic_locs = eigenvalues .< lambda_min
#         for i in 1:size(icas.h, 1), j in 1:size(icas.h, 2)
#             if problematic_locs[i, j] && i != j  # Exclude diagonal elements
#                 icas.h[i, j] += lambda_min - eigenvalues[i, j]
#             end
#         end
#     elseif mode == 1
#         icas.h .= icas.h + lambda_min  # Add lambda_min to all elements in the matrix
#     end
#     return icas.h
# end

# # Solve Hessian system
# function solveh(G::AbstractMatrix{T}, h::AbstractMatrix{T})  where T <: AbstractFloat
#     return (G .* h' .- G') ./ (h .* h' .- T(1))
# end

# # Line search
# function linesearch(ica::icastruct{T, M}, 
#         direction::AbstractMatrix{T}, 
#         initial_loss::Union{Nothing, T}=nothing, 
#         n_ls_tries::Int=10)  where {T <: AbstractFloat, M <: AbstractMatrix{T}}
#     m = size(ica.Y, 1)
#     mul!(ica.M_storage1, direction, ica.W)
    
#     step = T(1.0)
#     if initial_loss === nothing
#         initial_loss = loss(ica.Y, ica.W)
#     end
    
#     for n in 1:n_ls_tries
#         ica.M_storage2 .= I(m) .+ step .* direction
#         mul!(ica.M_storage4, ica.M_storage2, ica.Y)
#         new_W = ica.W .+ step .* ica.M_storage1
#         new_loss = loss(ica.M_storage4, new_W)
#         if new_loss > initial_loss
#             return true, ica.M_storage4, new_W, new_loss, step
#         end
#         step /= T(2.0)
#     end
#     return false, ica.Y, ica.W, initial_loss, step
# end

# # Generate data for testing
# function generate_data(m::Int, n::Int; use_gpu::Bool=false, use_metal::Bool=false)
#     S1 = create_array(randn(m, n), use_gpu, use_metal)
#     B = create_array(randn(m, m), use_gpu, use_metal)
#     return B * S1
# end


# using TimerOutputs

# function ica_ken(X::AbstractMatrix{T};
#                  maxiter::Int = 1000, 
#                  MM_iters::Int = 1000, 
#                  tol = 1e-6, 
#                  verbose::Bool = false, 
#                  W_warmStart::Union{AbstractMatrix{T}, Nothing} = nothing,
#                  nesterov::Bool = true) where {T <: AbstractFloat}

#     timer = TimerOutput()

#     use_metal = isa(X, MtlMatrix)
#     use_gpu = isa(X, CuArray)

#     if use_metal || use_gpu
#         println("using gpu")
#         tol = Float32(tol)
#         HALF = 0.5f0
#     else
#         tol = T(tol) 
#         HALF = T(0.5)
#         println("using cpu")
#     end

#     @timeit timer "icastruct" icas = icastruct(X, use_gpu, use_metal)
#     m, n = size(icas.X)
#     log_liks = create_array(zeros(T, 0), use_gpu, use_metal)

#     precon = Int(2) 
#     mem_size = Int(7) 
#     lambda_min = T(0.01)
#     ls_tries = Int(10)
#     gradient_norm = T(1)
#     current_loss = nothing

#     @timeit timer "init buffers" begin
#         icas.G_old .= similar(icas.W)
#         icas.G .= similar(icas.W)
#         W_old = similar(icas.W)
#         copyto!(W_old, icas.W)
#         W_prev = similar(icas.W)
#     end
#     t_k = T(1)
#     t_new = T(1)
#     if W_warmStart != nothing
#         copyto!(icas.W, W_warmStart)
#     end
#     mul!(icas.Y, icas.W, icas.X)

#     niters = maxiter
#     for iter in 1:maxiter
#         @timeit timer "score!" score!(icas)

#         if mod(iter, Int(5)) == 0
#             @timeit timer "gradient" icas.G .= gradient(icas)
#             @timeit timer "norm" gradient_norm = norm(icas.G, Inf)
#             if gradient_norm < tol
#                 niters = iter
#                 break
#             end
#         end

#         @timeit timer "SVD+updateW" begin
#             @. icas.M_storage4 = -icas.psiY + HALF * icas.Y
#             mul!(icas.M_storage2, icas.X, transpose(icas.M_storage4), HALF/n, T(0))
#             mul!(icas.M_storage1, transpose(icas.D_inv_sqrt), icas.M_storage2)
            
#             A_cpu = Matrix(icas.M_storage1)
#             S = svd(A_cpu)
#             copyto!(icas.M_storage1, create_array(S.U, use_gpu, use_metal))
#             copyto!(icas.M_storage2, create_array(S.V, use_gpu, use_metal))
#             copyto!(icas.E_storage, sqrt.(S.S .^ T(2) .+ T(1)) .+ S.S)
#             @. icas.M_storage2 = icas.E_storage * icas.M_storage2'
#             mul!(icas.M_storage3, icas.M_storage1, icas.M_storage2)

#             copyto!(icas.M_storage1, icas.W)
#             mul!(icas.W, icas.D_inv_sqrt, icas.M_storage3)
#             icas.W .= transpose(icas.W)
#         end

#         @timeit timer "nesterov+forward" begin
            
#             if nesterov
#                 t_new = (T(1) + sqrt(T(1) + T(4) * t_k^T(2))) / T(2)
#             else
#                 t_new = t_k
#             end
            
#             @timeit timer "nesterov1" θ = (t_k - 1) / t_new
#             @timeit timer "nesterov2" @. icas.M_storage1 = icas.W - W_old         # ΔW = W - W_old
#             @timeit timer "nesterov3" copyto!(W_old, icas.W)                      # W_old = W
#             @timeit timer "nesterov4" @. icas.W = icas.W + θ * icas.M_storage1    # fused update
#             @timeit timer "nesterov5" mul!(icas.Y, icas.W, icas.X)                # matrix multiply on GPU
#             t_k = t_new
            
            
#         end

#         if verbose
#             current_loss = loss(icas.Y, icas.W)
#             println("iteration ", iter, ", gradient norm: ", gradient_norm, 
#                 ", loglikelihood: ", current_loss)
#         end
#     end
    
#     final_loss = loss(icas.Y, icas.W)

#     println(timer)  # 输出所有步骤的耗时
#     return icas.W, niters, final_loss
# end


using LinearAlgebra, Random, Metal, CUDA, Distributions, LoopVectorization, AppleAccelerate

function create_array(data::Union{AbstractMatrix{T}, AbstractVector{T}}, use_gpu::Bool=false, use_metal::Bool=false) where T <: AbstractFloat
    if use_gpu
        @assert isdefined(Main, :CUDA) "CUDA.jl is not loaded. Please install and load CUDA.jl."
        return isa(data, CuArray) ? data : CuArray(data)
    elseif use_metal
        @assert isdefined(Main, :Metal) "Metal.jl is not loaded. Please install and load Metal.jl."
        return isa(data, MtlArray) ? data : MtlArray(data)
    else
        return isa(data, Array) ? data : Array(data)
    end
end

# Define the icastruct
struct icastruct{T <: AbstractFloat, A <: AbstractArray{T}}
    X :: A
    W :: A
    Y :: A
    D :: A
    D_inv :: A
    D_inv_sqrt :: A
    M_storage1 :: A
    M_storage2 :: A
    M_storage3 :: A
    M_storage4 :: A
    M_storage5 :: A
    Xt_storage :: A
    E_storage :: AbstractVector{T}
    G :: A
    G_old :: A
    psiY :: A
    psidY :: A
    direction :: A
    I_storage :: A
    h :: A
end

function icastruct(X::AbstractMatrix{T}, use_gpu::Bool=false, use_metal::Bool=false) where T <: AbstractFloat
    # Convert input matrix to the appropriate device (CPU/GPU)
    X_device = create_array(X, use_gpu, use_metal)
    m, n = size(X_device)
    @assert n ≥ m "Expect more samples (columns) than dimensions (rows) in X"
    
    if use_gpu || use_metal
        # perform in cpu
        X_cpu = Array(X_device)
        D_cpu = T(1) / n * (X_cpu * transpose(X_cpu))
        D_inv_cpu = inv(D_cpu)
        D_inv_sqrt_cpu = sqrt(D_inv_cpu)
        # back Metal
        D = create_array(D_cpu, use_gpu, use_metal)
        D_inv = create_array(D_inv_cpu, use_gpu, use_metal)
        D_inv_sqrt = create_array(D_inv_sqrt_cpu, use_gpu, use_metal)
    else
        D = T(1) / n * (X * transpose(X))
        D_inv = inv(D)
        D_inv_sqrt = sqrt(D_inv)
    end
    
    # Initialize other fields with consistent types
    W = create_array(Matrix{T}(I, m, m), use_gpu, use_metal)
    M_storage1 = create_array(zeros(T, m, m), use_gpu, use_metal)
    M_storage2 = create_array(zeros(T, m, m), use_gpu, use_metal)
    M_storage3 = create_array(zeros(T, m, m), use_gpu, use_metal)
    M_storage4 = create_array(zeros(T, m, n), use_gpu, use_metal)
    M_storage5 = create_array(zeros(T, m, n), use_gpu, use_metal)
    Xt_storage = create_array(zeros(T, n, m), use_gpu, use_metal)
    I_storage = create_array(Matrix{T}(I, m, m), use_gpu, use_metal)
    E_storage = create_array(zeros(T, m), use_gpu, use_metal)  # Reshape as 2D array
    
    Y = similar(X_device)
    copyto!(Y, X_device)

    # Return the icastruct instance
    return icastruct(X_device, 
        W, 
        Y,
        D, 
        D_inv, 
        D_inv_sqrt, 
        M_storage1, 
        M_storage2, 
        M_storage3, 
        M_storage4, 
        M_storage5, 
        Xt_storage, 
        E_storage,
        similar(W),
        similar(W),
        similar(X),
        similar(X),
        similar(W),
        I_storage,
        similar(W))
end

eltype(::icastruct{T, A}) where {T, A} = T



function score!(icas::icastruct{T, M}) where {T <: AbstractFloat, M <: AbstractMatrix{T}}
    if isa(icas.Y, CuArray) || occursin("Mtl", string(typeof(icas.Y)))
        icas.psiY .= tanh.(icas.Y ./ T(2))
    else
        @. icas.M_storage4 .= icas.Y ./ T(2)
        AppleAccelerate.tanh!(icas.psiY, icas.M_storage4)
    end
end

function loss(icas::icastruct{T, M}, Y::M, W::M) where {T <: AbstractFloat, M <: AbstractMatrix{T}}
    n = size(Y, 2)
#     log_det, _ = logabsdet(W)
    log_det = T(0)
    
    if isa(icas.Y, CuArray) || occursin("Mtl", string(typeof(icas.Y)))
        W_cpu = Array(W)
        log_det = log(abs(det(W_cpu)))
        icas.M_storage4 .= abs.(Y)
        sumY = sum(icas.M_storage4)
        @. icas.M_storage4 .= -icas.M_storage4
        icas.M_storage4 .= exp.(icas.M_storage4)
        icas.M_storage4 .= log1p.(icas.M_storage4)
        logcoshY = sumY + 2*sum(icas.M_storage4)
        
    else
        log_det = log(abs(det(W)))
        AppleAccelerate.abs!(icas.M_storage4, Y)
        sumY = sum(icas.M_storage4)
        @. icas.M_storage4 .= -icas.M_storage4

        AppleAccelerate.exp!(icas.M_storage4, icas.M_storage4)
        AppleAccelerate.log1p!(icas.M_storage4, icas.M_storage4)
        logcoshY = sumY + 2*sum(icas.M_storage4)
        
    end
    
    return log_det - logcoshY / n
end

# Gradient computation
function gradient(icas::icastruct{T, M}) where {T <: AbstractFloat, M <: AbstractMatrix{T}}
    m, n = size(icas.Y)
    
    copyto!(icas.M_storage1, icas.I_storage) #   icas.M_storage1 .= I(m)
    alpha = one(T) / T(n)
    beta = -one(T)
    mul!(icas.M_storage1, icas.psiY, transpose(icas.Y), alpha, beta)
    
    return icas.M_storage1
end


# Generate data for testing
function generate_data(m::Int, n::Int; use_gpu::Bool=false, use_metal::Bool=false)
    S1 = create_array(randn(m, n), use_gpu, use_metal)
    B = create_array(randn(m, m), use_gpu, use_metal)
    return B * S1
end


using TimerOutputs

function ica_ken(X::AbstractMatrix{T};
                 maxiter::Int = 1000, 
                 MM_iters::Int = 1000, 
                 tol = 1e-6, 
                 verbose::Bool = false, 
                 W_warmStart::Union{AbstractMatrix{T}, Nothing} = nothing,
                 nesterov::Bool = true) where {T <: AbstractFloat}

    timer = TimerOutput()

    use_metal = isa(X, MtlMatrix)
    use_gpu = isa(X, CuArray)

    if use_metal || use_gpu
        println("using gpu")
        tol = Float32(tol)
        HALF = 0.5f0
    else
        tol = T(tol) 
        HALF = T(0.5)
        println("using cpu")
    end

    @timeit timer "icastruct" icas = icastruct(X, use_gpu, use_metal)
    m, n = size(icas.X)
    log_liks = create_array(zeros(T, 0), use_gpu, use_metal)

    gradient_norm = T(1)
    current_loss = nothing
    final_loss = nothing
    
    @timeit timer "init buffers" begin
        icas.G_old .= similar(icas.W)
        icas.G .= similar(icas.W)
        W_old = similar(icas.W)
        copyto!(W_old, icas.W)
        W_prev = similar(icas.W)
    end
    t_k = T(1)
    t_new = T(1)
    if W_warmStart != nothing
        copyto!(icas.W, W_warmStart)
    end
    mul!(icas.Y, icas.W, icas.X)
    
    ck = 0.5f0
    
    cks = similar(icas.Y)

    A_cpu = Matrix(icas.M_storage1)
    
    niters = maxiter
    for iter in 1:maxiter
        @timeit timer "score!" score!(icas)
        
#         @timeit timer "ck1!" @. cks = icas.psiY / icas.Y 
#         @timeit timer "ck2!" ck = maximum(cks)
        
        
        if mod(iter, Int(5)) == 0
            @timeit timer "gradient" icas.G .= gradient(icas)
            @timeit timer "norm" gradient_norm = norm(icas.G, Inf)
            if gradient_norm < tol
                niters = iter
                break
            end
        end
        
        @timeit timer "SVD+updateW" begin
            @. icas.M_storage4 = icas.Y - icas.psiY/ck   # A
            mul!(icas.M_storage2, icas.X, transpose(icas.M_storage4), ck*HALF/n, T(0))  # C
            mul!(icas.M_storage1, icas.D_inv_sqrt, icas.M_storage2, 1/sqrt(ck) ,T(0))
            
            copyto!(A_cpu, icas.M_storage1)  # Ensure A_cpu is updated
            S = svd(A_cpu)
            copyto!(icas.M_storage1, create_array(S.U, use_gpu, use_metal))
            copyto!(icas.M_storage2, create_array(S.V, use_gpu, use_metal))
            copyto!(icas.E_storage, sqrt.(S.S .^ T(2) .+ T(1)) .+ S.S)
            
            Dia = Diagonal(icas.E_storage) 
            icas.M_storage2 .= icas.M_storage2 * Dia
            mul!(icas.M_storage3, icas.M_storage2, icas.M_storage1')

            mul!(icas.W, icas.M_storage3, icas.D_inv_sqrt, 1/sqrt(ck) ,T(0))
        end
        
        @timeit timer "nesterov+forward" begin
            
            if nesterov
                t_new = (T(1) + sqrt(T(1) + T(4) * t_k^T(2))) / T(2)
            else
                t_new = t_k
            end
            
            @timeit timer "nesterov1" θ = (t_k - 1) / t_new
            @timeit timer "nesterov2" @. icas.M_storage1 .= icas.W - W_old         # ΔW = W - W_old
            @timeit timer "nesterov3" copyto!(W_old, icas.W)                      # W_old = W
            @timeit timer "nesterov4" @. icas.W .= icas.W + θ * icas.M_storage1    # fused update
            @timeit timer "nesterov5" mul!(icas.Y, icas.W, icas.X)                # matrix multiply on GPU
            t_k = t_new
            
        end

        if verbose
            current_loss = loss(icas, icas.Y, icas.W)
            println("iteration ", iter, ", gradient norm: ", gradient_norm, 
                ", loglikelihood: ", current_loss)
        end
    end
    
    # final_loss = loss(icas, icas.Y, icas.W)

    println(timer)  # 输出所有步骤的耗时
    return icas.W, niters, final_loss
end

