
include("SPHsolver.jl")

using JLD2
using Dates: format, now

dx = 20e-6

R = 340e-6
L = 3R
w = 5dx

@show R/dx
@show R/w

s_ff = 2e-3
s_fs = [0.2, 0.5] .* s_ff

datetime = format(now(), "YY-mm-dd_HH-MM")
saveFolder = get(ENV, "OUTPUT_DIR", "./outputs/striped/$datetime")

Δtmin = 1e-12
Δtmax = 1e-5

maxiters = Inf
maxt = 0.020

saveΔt = Inf
plotΔt = 5e-4
printΔt = 1e-5

function stripedPF(r, h, label)
    if label === :fluid
        return myPolynomial11(r,h)
    else
        return myPolynomial16(r,h)
    end
end

# 50% glycerol
physics = PhysicalParameters{3,Float64}(
    g = SVector{3,Float64}(0, 0, -9.81),
    ρ₀ = 1261,
    c₀ = 80, # 10 √ (pmax / ρ₀)
    # α = 0.01,
    μ = 5.9e-3, # Pa.s
    δ = 0,
    viscous = true,
    s = Dict(
        :fluid => s_ff,
        :dry => s_fs[1],
        :wet => s_fs[2]
    ),
    F = stripedPF
)

PL = fluidSphere(dx, R, physics.ρ₀) *
    stripedChemically(dx, L, physics.ρ₀, w)

PL.position[PL.label .=== :fluid] .+= [[0, 0, R + 2dx]]

# Perturb positions slightly to kick-start rearrangement from surface tension
for i in 1:length(PL)
    if PL.label[i] === :fluid
        PL.position[i] += dx/20 * (rand(SVector{3,Float64}) .- 0.5)
    end
end

for i in eachParticle(PL)
    if PL.label[i] !== :fluid
        if abs(PL.position[i][2]) <= 8dx
            PL.label[i] = :wet
        else
            PL.label[i] = :dry
        end
    end
end

κ = 4.0 # radius of support multiplier
padFactor = 1.005

exclLabelLeft = (:dry, :wet)
exclLabelRight = ()
NL = constructNeighbourList!(PL, κ, padFactor; exclLabelLeft, exclLabelRight)

writeBoundaryVTK(PL, saveFolder; boundaryLabels = exclLabelLeft)

nf = count(isequal(:fluid), PL.label)
nb = count(!isequal(:fluid), PL.label)

@info "Using $(length(NL)) particles: $nf fluid and $nb boundary"

predictorCorrectorTimestepper!(NL, physics, Δtmin, Δtmax, maxiters, maxt, 
    saveΔt, plotΔt, printΔt; saveFolder, onlySaveSummaryData = true,
    exclLabelLeft, exclLabelRight)