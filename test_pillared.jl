
include("SPHsolver.jl")

using JLD2
using Dates: format, now

dx = 50e-6

R = 30dx
L = 80dx
ph = 5dx
pw = 4dx
pd = 8dx

s_ff = 2e-3
s_fs = 0.35 * s_ff

datetime = format(now(), "YY-mm-dd_HH-MM")
saveFolder = "./outputs/pillared/$datetime"

Δtmin = 1e-12
Δtmax = 1e-5

maxiters = Inf

saveΔt = Inf
plotΔt = 1e-4
printΔt = 1e-5

physics = PhysicalParameters{3,Float64}(
    g = SVector{3,Float64}(0, 0, -9.81),
    ρ₀ = 1000,
    c₀ = 80, # 10 √ (pmax / ρ₀)
    # α = 0.01,
    μ = 8.9e-4,
    δ = 0,
    viscous = true,
    s = Dict(
        :fluid => s_ff,
        :fixedGhost => s_fs
    ),
    F = pairwiseForceFunc_11_16
)

physics_zerog = PhysicalParameters{3,Float64}(
    g = SVector{3,Float64}(0, 0, 0),
    ρ₀ = 1000,
    c₀ = 80, # 10 √ (pmax / ρ₀)
    # α = 0.01,
    μ = 8.9e-4,
    δ = 0,
    viscous = true,
    s = Dict(
        :fluid => s_ff,
        :fixedGhost => s_fs
    ),
    F = pairwiseForceFunc_11_16
)

PL = fluidSphere(dx, R, physics.ρ₀) * pillaredSurface(dx, L, pd, pw, ph, physics.ρ₀, :fixedGhost)

PL.position[PL.label .=== :fluid] .+= [[0, 0, R + dx]]

# Perturb positions slightly to kick-start rearrangement from surface tension
for i in 1:length(PL)
    if PL.label[i] === :fluid
        PL.position[i] += dx/20 * (rand(SVector{3,Float64}) .- 0.5)
    end
end

κ = 4.0 # radius of support multiplier
padFactor = 1.0

exclLabelLeft = (:fixedGhost,)
exclLabelRight = ()
NL = constructNeighbourList!(PL, κ, padFactor; exclLabelLeft, exclLabelRight)

writeBoundaryVTK(PL, saveFolder; boundaryLabels = exclLabelLeft)

nf = count(isequal(:fluid), PL.label)
nb = count(!isequal(:fluid), PL.label)

@info "Using $(length(NL)) particles: $nf fluid and $nb boundary"

predictorCorrectorTimestepper!(NL, physics_zerog, Δtmin, Δtmax, maxiters, 0.01, 
    saveΔt, plotΔt, printΔt; saveFolder=joinpath(saveFolder, "0g"), 
    onlySaveSummaryData = true, exclLabelLeft, exclLabelRight)

predictorCorrectorTimestepper!(NL, physics, Δtmin, Δtmax, maxiters, 0.05, 
    saveΔt, plotΔt, printΔt; saveFolder=joinpath(saveFolder, "1g"), 
    onlySaveSummaryData = true, exclLabelLeft, exclLabelRight)