export infomax_ica

using LinearAlgebra, Statistics, Random

# ---------- Struct ----------
struct ICAState{T <: AbstractFloat}
    X::Matrix{T}
    W::Matrix{T}
    WX::Matrix{T}
    M_storage1::Matrix{T}
    M_storage2::Matrix{T}
end

function ICAState(X::Matrix{T}) where T <: AbstractFloat
    m, _ = size(X)
    W = Matrix{T}(I, m, m)
    WX = W*X
    M_storage1 = similar(W)
    M_storage2 = similar(W)
    return ICAState(X, W, WX, M_storage1, M_storage2)
end

# # ---------- Natural gradient ----------
function natural_gradient(WX::Matrix{T}, W::Matrix{T}, X::Matrix{T}, icas) where T
    m, n = size(X)
    copyto!(icas.M_storage1, I)
    mul!(icas.M_storage1, Gprime.(WX), transpose(WX), one(T)/T(n), one(T))
    mul!(icas.M_storage2, icas.M_storage1, W)
    return icas.M_storage2
end

# ---------- Main algorithm ----------
function infomax_ica(X::Matrix{T};
        maxiter::Int = 200,
        tol::T = 1e-6,
        η::T = 1.0,
        verbose::Bool = true) where T <: AbstractFloat
    
    icas = ICAState(X)
    m, n = size(X)
    log_old = log_likelihood(icas.WX, icas.W, icas.X)
    niters = maxiter

    for iter in 1:maxiter
        icas.M_storage1 .= natural_gradient(icas.WX, icas.W, icas.X, icas)
        
        # Check convergence
        if norm(icas.M_storage1) / m^2 < tol
#             verbose && println("Converged at iteration $iter")
            niters = iter
            break
        end

        # update W
        copy!(icas.M_storage2, icas.W)
        LinearAlgebra.axpy!(T(η), icas.M_storage1, icas.M_storage2)
        
#         icas.M_storage2 .= icas.W + η * icas.M_storage1
        mul!(icas.WX, icas.M_storage2, icas.X)
        
#         WX_new = W_new * X
        log_new = log_likelihood(icas.WX, icas.M_storage2, icas.X)

        # Backtracking line search
        while log_new < log_old
            η *= 0.5
            copy!(icas.M_storage2, icas.W)
            LinearAlgebra.axpy!(T(η), icas.M_storage1, icas.M_storage2)
            mul!(icas.WX, icas.M_storage2, icas.X)
            log_new = log_likelihood(icas.WX, icas.M_storage2, icas.X)
        end

        icas.W .= icas.M_storage2
        log_old = log_new
    end

    return icas.W, niters, log_old
end

# end # module


