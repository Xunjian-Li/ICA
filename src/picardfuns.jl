export picard

using LinearAlgebra, Random, Distributions, LoopVectorization, TimerOutputs, AppleAccelerate

struct picardstruct{T <: AbstractFloat, M <: AbstractMatrix{T}}
    X::M
    W::M
    Y::M
    M_storage1 ::M
    M_storage2 ::M
    M_storage3 ::M
    M_storage4 ::M
    M_storage5 ::M
    G::M
    G_old::M
    direction::M
    psiY::M
    psidY::M
    h::M
end

function picardstruct(X::AbstractMatrix{T}) where T <: AbstractFloat
    m, n = size(X)
    W = Matrix{T}(I, m, m)
    return picardstruct(
        X,
        W,
        copy(X),
        similar(W),
        similar(W),
        similar(X),
        similar(X),
        similar(X),
        similar(W),
        similar(W),
        similar(W),
        similar(X),
        similar(X),
        Matrix{T}(undef, m, m)
    )
end

eltype(::picardstruct{T, A}) where {T, A} = T

# Helper function to compute the L-BFGS direction
function _l_bfgs_direction1(ica::picardstruct{T, M}, s_list, y_list, r_list, precon, lambda_min)  where {T <: AbstractFloat, M <: AbstractMatrix{T}}
    q = copy(ica.G)
    a_list = T[]
    for (s, y, r) in zip(reverse(s_list), reverse(y_list), reverse(r_list))
        alpha = r * sum(s .* q)
        push!(a_list, alpha)
        q .-= alpha .* y
    end
    z = solveh1(q, ica.h)
    
    for (s, y, r, alpha) in zip(s_list, y_list, r_list, reverse(a_list))
        beta = r * sum(y .* z)
        z .+= (alpha - beta) .* s
    end
    
    return -z
end

function score1!(ica::picardstruct{T, M}) where {T <: AbstractFloat, M <: AbstractMatrix{T}}
    @. ica.M_storage3 .= ica.Y ./ T(2)
    AppleAccelerate.tanh!(ica.psiY, ica.M_storage3)
end

function score_der1!(Y::AbstractMatrix{T}, X::AbstractMatrix{T}) where {T <: AbstractFloat}
    @assert size(Y) == size(X) "X and Y must have the same size"
    @turbo for j in axes(X, 2), i in axes(X, 1)
        Y[i, j] = (1 - X[i, j]^2) / 2
    end
end

function loss1(ica::picardstruct{T, M}, Y::M, W::M) where {T <: AbstractFloat, M <: AbstractMatrix{T}}
    n = size(Y, 2)
#     log_det, _ = logabsdet(W)
    log_det = log(abs(det(W)))
    
    AppleAccelerate.abs!(ica.M_storage4, Y)
    sumY = sum(ica.M_storage4)
    @. ica.M_storage4 .= -ica.M_storage4
    
    AppleAccelerate.exp!(ica.M_storage4, ica.M_storage4)
    AppleAccelerate.log1p!(ica.M_storage4, ica.M_storage4)
    logcoshY = sumY + 2*sum(ica.M_storage4)

    return -log_det + logcoshY / n
end

# Gradient computation
function gradient1(ica::picardstruct{T, M}) where {T <: AbstractFloat, M <: AbstractMatrix{T}}
    m, n = size(ica.Y)

    ica.M_storage1 .= Matrix{T}(I, m, m)
    mul!(ica.M_storage1, ica.psiY, transpose(ica.Y), T(1)/n, -T(1))
    
    return ica.M_storage1
end

# Compute Hessian approximation
function compute_h1(ica::picardstruct{T, M}, precon::Int)  where {T <: AbstractFloat, M <: AbstractMatrix{T}}
    m, n = size(ica.Y)
    if precon == 2
        @. ica.M_storage3 .= ica.Y ^2
        mul!(ica.M_storage1, ica.psidY, transpose(ica.M_storage3), T(1)/n, T(0))
        return ica.M_storage1
    else
        @. ica.M_storage4 .= ica.Y .^2
        sigma2 = mean(ica.M_storage4, dims=2)
        psidY_mean = mean(ica.psidY, dims=2)
        @. ica.h .= psidY_mean .* sigma2'
        diagonal_term = mean(ica.M_storage4)
        for i in 1:m
            ica.h[i, i] .= diagonal_term
        end
        return ica.h
    end
end

function compute_eigenvalues1!(eigenvalues, h::M) where {T <: AbstractFloat, M <: AbstractMatrix{T}}
    n, m = size(h)
    @inbounds for i in 1:n
        @simd for j in 1:m
            a = h[i, j]
            b = h[j, i]
            diff = a - b
            sum_ = a + b
            sq = diff^2 + T(4.0)
            eigenvalues[i, j] = T(0.5) * (sum_ - sqrt(sq))
        end
    end
    return eigenvalues
end

function regularize_h1(ica::picardstruct{T, M}, lambda_min::T, mode::Int=0)  where {T <: AbstractFloat, M <: AbstractMatrix{T}}
    """
    Regularizes the Hessian approximation `h` using the constant `lambda_min`.
    Mode selects the regularization algorithm:
    0 -> Shift each eigenvalue below `lambda_min` to `lambda_min`.
    1 -> Add `lambda_min * I` to `h`.
    """
    if mode == 0
        
        eigenvalues = similar(ica.h)
        compute_eigenvalues1!(eigenvalues, ica.h)
        
        # Regularize
        problematic_locs = eigenvalues .< lambda_min
        for i in 1:size(ica.h, 1), j in 1:size(ica.h, 2)
            if problematic_locs[i, j] && i != j  # Exclude diagonal elements
                ica.h[i, j] += lambda_min - eigenvalues[i, j]
            end
        end
    elseif mode == 1
        ica.h .= ica.h + lambda_min  # Add lambda_min to all elements in the matrix
    end
    return ica.h
end

# Solve Hessian system
function solveh1(G::M, h::M) where {T <: AbstractFloat, M <: AbstractMatrix{T}}
    return (G .* h' .- G') ./ (h .* h' .- T(1))
end

# Line search
function linesearch1(ica::picardstruct{T, M}, 
        direction::M, 
        initial_loss::Union{Nothing, T}=nothing, 
        n_ls_tries::Int=10)  where {T <: AbstractFloat, M <: AbstractMatrix{T}}
    m = size(ica.Y, 1)
    mul!(ica.M_storage1, direction, ica.W)
    
    step = T(1)
    if initial_loss === nothing
        initial_loss = loss1(ica, ica.Y, ica.W)
    end
    
    for n in 1:n_ls_tries
        ica.M_storage2 .= Matrix{T}(I, m, m) .+ step .* direction
        mul!(ica.M_storage3, ica.M_storage2, ica.Y)
        
        new_W = ica.W .+ step .* ica.M_storage1
        
        new_loss = loss1(ica, ica.M_storage3, new_W)
        
        if new_loss < initial_loss
            return true, ica.M_storage3, new_W, new_loss, step
        end
        step /= T(2)
    end
    return false, ica.Y, ica.W, initial_loss, step
end


function picard(
    X::Matrix{T}; 
    max_iter::Int=1000,
    tol = 1e-6,
    mem_size::Int=7,
    precon::Int=2,
    lambda_min::T=T(0.01),
    infomax::Bool=false,
    ls_tries::Int=10,
    verbose::Bool=false,
    callback::Union{Function, Nothing}=nothing
    ) where T <: AbstractFloat

    timer = TimerOutput()

    @timeit timer "initialize" begin
        ica = picardstruct(X)
        tol = T(tol)
        m, n = size(ica.X)
        ica.Y .= copy(ica.X)
        s_list = Matrix{T}[]
        y_list = Matrix{T}[]
        r_list = T[]
        converged = false
        current_loss = nothing
        direction = similar(ica.W)
    end

    n_iters = max_iter

    for n_iter in 1:max_iter
        @timeit timer "score + gradient" begin
            @timeit timer "score" score1!(ica)
            if !infomax
                score_der1!(ica.psidY, ica.psiY)
            end
            @timeit timer "gradient" ica.G .= gradient1(ica)
        end

        @timeit timer "gradient norm + check" begin
            gradient_norm = norm(ica.G, Inf)
            if gradient_norm < tol
                n_iters = n_iter
                break
            end
        end

        @timeit timer "L-BFGS & linesearch" begin
            if !infomax
                if n_iter > 1
                    push!(s_list, direction)
                    y = ica.G - ica.G_old
                    push!(y_list, y)
                    push!(r_list, T(1) / dot(vec(direction), vec(y)))
                    if length(s_list) > mem_size
                        popfirst!(s_list); popfirst!(y_list); popfirst!(r_list)
                    end
                end
                copyto!(ica.G_old, ica.G)
                ica.h .= compute_h1(ica, precon)
                ica.h .= regularize_h1(ica, lambda_min)
                
                println("ica.h: ", ica.h)
                
                direction = _l_bfgs_direction1(ica, s_list, y_list, r_list, precon, lambda_min)
                converged, new_Y, new_W, new_loss, alpha = linesearch1(ica, direction, current_loss, ls_tries)
            end
            
            if !converged || infomax
                direction = -ica.G
                s_list, y_list, r_list = Matrix{T}[], Matrix{T}[], T[]
                _, new_Y, new_W, new_loss, alpha = linesearch1(ica, direction, current_loss, ls_tries)
            end
        end
        
        @timeit timer "update" begin
            direction .*= alpha
            ica.Y .= new_Y
            ica.W .= new_W
            current_loss = new_loss
        end
        
        if verbose
            println("Iteration $n_iter, ∥grad∥ = $(round(gradient_norm, digits=6)), Loss = $(round(-current_loss, digits=6))")
        end
        
        if callback !== nothing
            callback(; Y=ica.Y, W=ica.W, G=ica.G, n=n_iter)
        end
    end

    show(timer)

    return ica.Y, ica.W, -current_loss, n_iters, timer
end


