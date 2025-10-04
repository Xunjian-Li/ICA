export tanh_accel

"""
    tanh_accel(y, x)

In-place apply `tanh` using Apple's Accelerate framework. 
Supports `Vector{Float64}` and `Vector{Float32}`.
`y` and `x` must have the same length.
"""
function tanh_accel!(y::Vector{Float64}, x::Vector{Float64})
    n = length(x)
    @assert length(y) == n
    ccall((:vvtanh, "libAccelerate"), Cvoid,
        (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int32}),
        y, x, Ref(Int32(n)))
    return y
end

function tanh_accel(y::Vector{Float32}, x::Vector{Float32})
    n = length(x)
    @assert length(y) == n
    ccall((:vvtanhf, "libAccelerate"), Cvoid,
        (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Int32}),
        y, x, Ref(Int32(n)))
    return y
end
