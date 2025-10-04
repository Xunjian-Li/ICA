versioninfo()

using Pkg

Pkg.activate(pwd())
Pkg.instantiate()
Pkg.status()

using LinearAlgebra, Random, Plots, BenchmarkTools, MultivariateStats
using WAV, Metal, MAT, Statistics

using ICAmm
using Distributions, LoopVectorization
using TimerOutputs

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

# using GPUArraysCore

# # Disallow scalar indexing globally
# allowscalar(true)

tol = 1e-4

X_whitened_GPU = MtlArray(Float32.(X_whitened))

X_whitened_CPU = Float32.(X_whitened)

@btime W, niters, final_loss, Y = ica_ken(X_whitened_CPU,
    maxiter=500,
    nesterov=true,
    tol=tol,
    verbose=false);

@btime W, niters, final_loss, Y = ica_ken(X_whitened_GPU,
    maxiter=500,
    nesterov=true,
    tol=tol,
    verbose=false);