
using LinearAlgebra, Random, Metal, CUDA, Distributions, LoopVectorization, AppleAccelerate

# Define the icastruct
struct icastruct{T <: AbstractFloat, A <: AbstractArray{T}}
    X :: A
    W :: A
    Y :: A
    D_inv_sqrt :: A
    M_storage1 :: A
    M_storage2 :: A
    M_storage3 :: A
    M_storage4 :: A
    M_storage5 :: A
    E_storage :: AbstractVector{T}
    G :: A
    psiY :: A
    psidY :: A
    direction :: A
    I_storage :: A
    h :: A
end

function icastruct(X::AbstractMatrix{T}) where T <: AbstractFloat
    
    m, n = size(X)
    @assert n ≥ m "Expect more samples (columns) than dimensions (rows) in X"
    
    D = T(1) / n * (X * transpose(X))
    D_inv = inv(D)
    D_inv_sqrt = sqrt(D_inv)
    
    # Initialize other fields with consistent types
    W = Matrix{T}(I, m, m)
    M_storage1 = zeros(T, m, m)
    M_storage2 = zeros(T, m, m)
    M_storage3 = zeros(T, m, m)
    M_storage4 = zeros(T, m, n)
    M_storage5 = zeros(T, m, n)
    I_storage = Matrix{T}(I, m, m)
    E_storage = zeros(T, m)
    
    Y = similar(X)
    Y .= 0

    # Return the icastruct instance
    return icastruct(X, 
        W,
        Y,
        D_inv_sqrt, 
        M_storage1, 
        M_storage2, 
        M_storage3, 
        M_storage4, 
        M_storage5, 
        E_storage,
        similar(W),
        similar(X),
        similar(X),
        similar(W),
        I_storage,
        similar(W))
end

eltype(::icastruct{T, A}) where {T, A} = T

function threaded_tanh!(Z::AbstractMatrix, Y::AbstractMatrix)
    @threads for j in 1:size(Y, 2)
        @turbo for i in 1:size(Y, 1)
            Z[i, j] = tanh(Y[i, j]/2)
        end
    end
end

function score!(icas::icastruct{T, M}) where {T <: AbstractFloat, M <: AbstractMatrix{T}}
    threaded_tanh!(icas.psiY, icas.Y)
end

function loss(icas::icastruct{T, M}, Y::M, W::M) where {T <: AbstractFloat, M <: AbstractMatrix{T}}
    n = size(Y, 2)
    log_det = T(0)
    log_det = log(abs(det(W)))
    AppleAccelerate.abs!(icas.M_storage4, Y)
    sumY = sum(icas.M_storage4)
    @. icas.M_storage4 .= -icas.M_storage4
    AppleAccelerate.exp!(icas.M_storage4, icas.M_storage4)
    AppleAccelerate.log1p!(icas.M_storage4, icas.M_storage4)
    logcoshY = sumY + 2*sum(icas.M_storage4)
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
    S1 = randn(m, n)
    B  = randn(m, m)
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
    
    tol = T(tol) 
    HALF = T(0.5)
    println("using cpu")
    
    @timeit timer "icastruct" icas = icastruct(X)
    m, n = size(icas.X)
    log_liks = zeros(T, 0)

    gradient_norm = T(1)
    current_loss = nothing
    final_loss = nothing
    
    @timeit timer "init buffers" begin
        icas.G .= similar(icas.W)
        W_old = similar(icas.W)
        copyto!(W_old, icas.W)
    end
    t_k = T(1)
    t_new = T(1)
    
    if W_warmStart != nothing
        copyto!(icas.W, W_warmStart)
    end
    
    mul!(icas.Y, icas.W, icas.X)
    
    niters = maxiter
    for iter in 1:maxiter
        
        @timeit timer "score!" score!(icas)
        
        if mod(iter, Int(5)) == 0
            @timeit timer "gradient" icas.G .= gradient(icas)
            @timeit timer "norm" gradient_norm = norm(icas.G, Inf)
            if gradient_norm < tol
                niters = iter
                break
            end
        end
        
        @timeit timer "SVD+updateW" begin
            @timeit timer "addition" @turbo for i in eachindex(icas.M_storage4)
                        icas.M_storage4[i] = HALF * icas.Y[i] - icas.psiY[i]
                    end
            
            @timeit timer "multiplication" mul!(icas.M_storage2, icas.X, transpose(icas.M_storage4))  # C
            @timeit timer "updateW" mul!(icas.M_storage1, icas.D_inv_sqrt, icas.M_storage2, sqrt(HALF)/n ,T(0))
            
            @timeit timer "SVD" S = svd(icas.M_storage1)
            @timeit timer "updateW" copyto!(icas.E_storage, sqrt.(S.S .^ T(2) .+ T(1)) .+ S.S)
            
            @timeit timer "updateW" Dia = Diagonal(icas.E_storage) 
            @timeit timer "updateW" mul!(icas.M_storage2, S.V, Dia)
            @timeit timer "updateW" mul!(icas.M_storage3, icas.M_storage2, transpose(S.U))
            @timeit timer "updateW" mul!(icas.W, icas.M_storage3, icas.D_inv_sqrt, 1/sqrt(HALF) ,T(0))
        end
        
        @timeit timer "nesterov+forward" begin
            
            if nesterov
                t_new = (T(1) + sqrt(T(1) + T(4) * t_k^T(2))) / T(2)
            else
                t_new = t_k
            end
            
            @timeit timer "nesterov" θ = (t_k - 1) / t_new
            @timeit timer "nesterov" @. icas.M_storage1 .= icas.W - W_old         # ΔW = W - W_old
            @timeit timer "nesterov" copyto!(W_old, icas.W)                      # W_old = W
            @timeit timer "nesterov" @. icas.W .= icas.W + θ * icas.M_storage1    # fused update
            @timeit timer "update Y" mul!(icas.Y, icas.W, icas.X)                # matrix multiply on GPU
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
