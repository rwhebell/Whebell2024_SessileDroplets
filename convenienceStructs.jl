using StaticArrays

struct PhysicalParameters{dim,T}

    g::SVector{dim,T}                   # acceleration, gravity
    ρ₀::T                               # reference density
    μ::T                                # dynamic viscosity
    c₀::T                               # numerical speed of sound
    α::T                                # artificial viscosity
    δ::T                                # numerical density diffusion
    viscous::Bool                       # switch true viscosity on/off
    s::Dict{Symbol, T}                  # hash table of surface tension strengths
    F::Function                         # pairwise force profiles

    function PhysicalParameters{dim,T}(;
        g = zeros(dim),
        ρ₀ = NaN,
        μ = NaN,
        c₀ = NaN,
        α = NaN,
        δ = NaN,
        viscous = true,
        s = Dict(),
        F = ((_,_,_) -> 0)
    )  where {dim,T}

        @assert (viscous && !isnan(μ)) || (!viscous && !isnan(α))
        
        return new{dim,T}(g, ρ₀, μ, c₀, α, δ, viscous, s, F)

    end

end