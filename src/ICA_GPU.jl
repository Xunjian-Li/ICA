
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
struct icastruct{T <: AbstractFloat, A <: AbstractArray{T}, B <: AbstractMatrix{T}}
    X :: A
    X_CPU :: B
    W :: A
    Y :: A
    D_inv_sqrt :: B
    A_cpu :: B
    B_cpu :: B
    M_storage1 :: A
    M_storage2 :: A
    M_storage3 :: A
    M_storage4 :: A
    M_storage5 :: B
    E_storage :: AbstractVector{T}
    G :: A
    psiY :: A
    psidY :: A
    direction :: A
    I_storage :: A
    h :: A
end

function icastruct(X::AbstractMatrix{T}, use_gpu::Bool=false, use_metal::Bool=false) where T <: AbstractFloat
    # Convert input matrix to the appropriate device (CPU/GPU)
    X_device = create_array(X, use_gpu, use_metal)
    X_CPU = create_array(X)
    m, n = size(X_device)
    @assert n ≥ m "Expect more samples (columns) than dimensions (rows) in X"
    

    X_cpu = Array(X_device)
    D = T(1) / n * (X_cpu * transpose(X_cpu))
    D_inv = inv(D)
    D_inv_sqrt = sqrt(D_inv)
    A_cpu = similar(D_inv_sqrt)
    A_cpu .= 0
    B_cpu = similar(D_inv_sqrt)
    B_cpu .= 0
    
    # Initialize other fields with consistent types
    W = create_array(Matrix{T}(I, m, m), use_gpu, use_metal)
    M_storage1 = create_array(zeros(T, m, m), use_gpu, use_metal)
    M_storage2 = create_array(zeros(T, m, m), use_gpu, use_metal)
    M_storage3 = create_array(zeros(T, m, m), use_gpu, use_metal)
    M_storage4 = create_array(zeros(T, m, n), use_gpu, use_metal)
    M_storage5 = zeros(T, m, n)
    I_storage = create_array(Matrix{T}(I, m, m), use_gpu, use_metal)
    E_storage = zeros(T, m)
    
    Y = similar(X_device)
    Y .= 0

    # Return the icastruct instance
    return icastruct(
        X_device, 
        X_CPU,
        W, 
        Y,
        D_inv_sqrt,
        A_cpu,
        B_cpu,
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

eltype(::icastruct{T, A, B}) where {T, A, B} = T

function score!(icas::icastruct{T, M}) where {T <: AbstractFloat, M <: AbstractMatrix{T}}
    icas.psiY .= tanh.(icas.Y ./ T(2))
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

    use_metal = isa(X, MtlMatrix)
    use_gpu = isa(X, CuArray)

    timer = TimerOutput()

    println("using gpu")
    tol = Float32(tol)
    HALF = 0.5f0

    @timeit timer "icastruct" icas = icastruct(X, use_gpu, use_metal)
    m, n = size(icas.X)
    log_liks = create_array(zeros(T, 0), use_gpu, use_metal)

    gradient_norm = T(1)
    current_loss = nothing
    final_loss = nothing
    
    @timeit timer "init buffers" begin
        icas.G .= similar(icas.W)
        W_old = similar(icas.A_cpu)
        copyto!(W_old, icas.W)
        W_prev = similar(icas.W)
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
            @timeit timer "addition" icas.M_storage4 .= HALF * icas.Y - icas.psiY
            @timeit timer "multiplication" mul!(icas.M_storage2, icas.X, transpose(icas.M_storage4))  # C
            
            
            # calculate with cpu in low dimensional data
            @timeit timer "copyW" icas.A_cpu .= Array(icas.M_storage2)
            
            @timeit timer "updateW" mul!(icas.B_cpu, icas.D_inv_sqrt, icas.A_cpu, sqrt(HALF)/n ,T(0))
            @timeit timer "SVD" S = svd(icas.B_cpu)
            @timeit timer "updateW" copyto!(icas.E_storage, sqrt.(S.S .^ T(2) .+ T(1)) .+ S.S)
            
            @timeit timer "updateW" Dia = Diagonal(icas.E_storage) 
            @timeit timer "updateW" mul!(icas.A_cpu, S.V, Dia)
            @timeit timer "updateW" mul!(icas.B_cpu, icas.A_cpu, transpose(S.U))

            @timeit timer "updateW" mul!(icas.A_cpu, icas.B_cpu, icas.D_inv_sqrt, 1/sqrt(HALF) ,T(0))
            
#             @timeit timer "copyW" copyto!(icas.W, icas.A_cpu)
            
        end
        
        
        
        @timeit timer "nesterov+forward" begin
            
            if nesterov
                t_new = (T(1) + sqrt(T(1) + T(4) * t_k^T(2))) / T(2)
            else
                t_new = t_k
            end
            
            @timeit timer "nesterov" θ = (t_k - 1) / t_new
            
            @timeit timer "nesterov"  icas.B_cpu .= icas.A_cpu .- W_old         # ΔW = W - W_old
            
            @timeit timer "nesterov" copyto!(W_old, icas.A_cpu)                      # W_old = W
            @timeit timer "nesterov" @. icas.A_cpu .= icas.A_cpu + θ * icas.B_cpu    # fused update
            
            @timeit timer "update Y" mul!(icas.M_storage5, icas.A_cpu, icas.X_CPU)                # matrix multiply on GPU
            
            @timeit timer "copyto Y" icas.Y .= MtlArray(icas.M_storage5)
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
