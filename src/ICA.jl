versioninfo()

using Pkg

Pkg.activate(pwd())
Pkg.instantiate()
Pkg.status()

using BenchmarkTools
using Metal, MAT, Statistics, LinearAlgebra
# using ICAmm
using TimerOutputs


using MAT


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
    W :: A
    Y :: A
    D_inv_sqrt :: B
    M_storage1 :: A
    M_storage2 :: A
    M_storage3 :: A
    M_storage4 :: A
    M_storage5 :: A
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
        D = T(1) / n * (X_cpu * transpose(X_cpu))
        # D_inv_cpu = inv(D_cpu)
        # D_inv_sqrt_cpu = sqrt(D_inv_cpu)
        # back Metal
        # D = create_array(D_cpu, use_gpu, use_metal)
        # D_inv = create_array(D_inv_cpu, use_gpu, use_metal)
        # D_inv_sqrt = create_array(D_inv_sqrt_cpu, use_gpu, use_metal)
    else
        D = T(1) / n * (X * transpose(X))
    end

    D_inv = inv(D)
    D_inv_sqrt = sqrt(D_inv)
    
    # Initialize other fields with consistent types
    W = create_array(Matrix{T}(I, m, m), use_gpu, use_metal)
    M_storage1 = create_array(zeros(T, m, m), use_gpu, use_metal)
    M_storage2 = create_array(zeros(T, m, m), use_gpu, use_metal)
    M_storage3 = create_array(zeros(T, m, m), use_gpu, use_metal)
    M_storage4 = create_array(zeros(T, m, n), use_gpu, use_metal)
    M_storage5 = create_array(zeros(T, m, n), use_gpu, use_metal)
    I_storage = create_array(Matrix{T}(I, m, m), use_gpu, use_metal)
    E_storage = create_array(zeros(T, m), use_gpu, use_metal)  # Reshape as 2D array
    
    Y = similar(X_device)
    Y .= 0

    # Return the icastruct instance
    return icastruct(X_device, 
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
        similar(W),
        similar(X),
        similar(X),
        similar(W),
        I_storage,
        similar(W))
end

eltype(::icastruct{T, A, B}) where {T, A, B} = T



function score!(icas::icastruct{T, M, B}) where {T <: AbstractFloat, M <: AbstractMatrix{T}, B <: AbstractMatrix{T}}
    if isa(icas.Y, CuArray) || occursin("Mtl", string(typeof(icas.Y)))
        icas.psiY .= tanh.(icas.Y ./ T(2))
    else
        @. icas.M_storage4 .= icas.Y ./ T(2)
        AppleAccelerate.tanh!(icas.psiY, icas.M_storage4)
    end
end

function loss(icas::icastruct{T, M, B}, Y::M, W::M) where {T <: AbstractFloat, M <: AbstractMatrix{T}, B <: AbstractMatrix{T}}
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
function gradient(icas::icastruct{T, M, B}) where {T <: AbstractFloat, M <: AbstractMatrix{T}, B <: AbstractMatrix{T}}
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

    M_storage1 = Float32.(zeros(m,m))
    M_storage2 = similar(M_storage1)
    M_storage3 = similar(M_storage1)
    W_new = similar(M_storage1)
    
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
            @timeit timer "SVD1" @. icas.M_storage4 = HALF .* icas.Y - icas.psiY   # A
            
            @timeit timer "SVD2" Metal.@sync mul!(icas.M_storage2, icas.X, transpose(icas.M_storage4), HALF/n, T(0))  # C
            @timeit timer "SVD3" copyto!(M_storage2, icas.M_storage2)
            @timeit timer "SVD4" mul!(M_storage1, icas.D_inv_sqrt, M_storage2, 1/sqrt(HALF) ,T(0))
            @timeit timer "SVD5" S = svd(M_storage1)
            @timeit timer "SVD6" copyto!(M_storage1, S.U)
            @timeit timer "SVD7" copyto!(M_storage2, S.V)
#             @timeit timer "SVD" # copyto!(icas.E_storage, sqrt.(S.S .^ T(2) .+ T(1)) .+ S.S)
            @timeit timer "SVD8" E_storage = sqrt.(S.S .^ T(2) .+ T(1)) .+ S.S
            @timeit timer "SVD9" Dia = Diagonal(E_storage)
            
            @timeit timer "SVD10" M_storage2 =  M_storage2 * Dia
            @timeit timer "SVD11" mul!(M_storage3, M_storage2, M_storage1')
            # mul!(icas.M_storage3, icas.M_storage2, icas.M_storage1')

            @timeit timer "SVD12" mul!(W_new, M_storage3, icas.D_inv_sqrt, 1/sqrt(HALF) ,T(0))
            @timeit timer "SVD13" copyto!(icas.W, W_new)

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
            @timeit timer "nesterov5" Metal.@sync mul!(icas.Y, icas.W, icas.X)                # matrix multiply on GPU
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


filename = "/Users/caleblee/codes/python/faster-ica/examples/eeg.mat"

file = matopen(filename)
X = read(file, "X")
close(file)

X_mean = mean(X, dims=2)
X = X .- X_mean


function whitening(X)
    cov_X = cov(X', corrected=false)
    eigenvalues, eigenvectors = eigen(cov_X)
    D_inv_sqrt = Diagonal(1 ./ sqrt.(eigenvalues))
    W = eigenvectors * D_inv_sqrt * eigenvectors'
    X_whitened = W * X
    return X_whitened, W
end

X_whitened, _ = whitening(X)

n_features, n_samples = size(X_whitened)

println("size :","(", n_features, " ", n_samples, ")")

tol = 1e-4

X_whitened_GPU = MtlArray(Float32.(X_whitened))

ica_ken(X_whitened_GPU,
    maxiter=500,
    nesterov=true,
    tol=tol,
    verbose=false);


