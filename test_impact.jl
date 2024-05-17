# using Pkg
# Pkg.activate(".")
# Pkg.resolve()
# Pkg.instantiate()

include("SPHsolver.jl")

BLAS.set_num_threads(1)

import Dates: format, now
using JLD2

##

dx = parse(Float64, get(ENV, "SPH_DX", "5e-5"))

R = parse(Float64, get(ENV, "SPH_DROP_RADIUS", "1e-3"))

s_ff = parse(Float64, get(ENV, "SPH_S_FF", "2.5e-3"))
s_fs = parse(Float64, get(ENV, "SPH_S_FS", "1.0e-3"))

N = parse(Int64, get(ENV, "SPH_NUM_V", "10"))

maxt = parse(Float64, get(ENV, "SPH_MAXT", "0.0001"))

arrayIndex = parse(Int, get(ENV, "PBS_ARRAY_INDEX", "3"))

impactVelocity = exp( range(log(0.1), log(3.0), length=N)[arrayIndex] )

pairwiseForceSymbol = get(ENV, "SPH_PAIRWISE_FORCE", "pairwiseForceFunc_11_16") |> Symbol
pairwiseForce = getfield(Main, pairwiseForceSymbol)

datetime = format(now(), "YY-mm-dd_HH-MM")
wholeJobID = replace( get(ENV, "PBS_JOBID", "fakeJob_$datetime"), r"\[\d{1,}\]" => "") # strip array index
saveDir = "./outputs/impact/" * wholeJobID
saveName = wholeJobID * "_" * lpad(arrayIndex, 3, "0")
saveFolderDefault = joinpath(saveDir, saveName)

saveFolder = get(ENV, "OUTPUT_DIR", saveFolderDefault)


##

function droplet_Height_Width_Positions(PL)

    coordsMatrix = reinterpret(reshape, Float64, PL.position)

    heightExtrema = extrema(coordsMatrix[3, PL.label .=== :fluid])
    
    radialDist = sqrt.( coordsMatrix[1, PL.label .=== :fluid].^2 .+ coordsMatrix[2, PL.label .=== :fluid].^2 )

    return (height = heightExtrema[2] - heightExtrema[1], 
        width = 2 * maximum(radialDist),
        fluidCoords = PL.position)

end

Δtmin = 1e-12
Δtmax = 1e-5

maxiters = Inf

saveΔt = 1e-5
printΔt = 1e-4
plotΔt = 1e-4

# https://doi.org/10.1016/S0009-2509(01)00175-0 (JCW, dynamic surface tension impacts)
# 50% glycerol solution
G50_physics = PhysicalParameters{3, Float64}(
    g = SVector{3,Float64}(0, 0, -9.81),
    ρ₀ = 1126,
    c₀ = 200, # 10 √ (pmax / ρ₀)
    # α = 0.01,
    μ = 5.9e-3,
    δ = 0,
    viscous = true,
    s = Dict(
        :fluid => s_ff,
        :fixedGhost => s_fs
    ),
    F = pairwiseForce
)

physics = G50_physics

PL = r_mm_drop(physics.ρ₀, R, dx; xy_extent=4R)

# Perturb positions slightly to kick-start rearrangement from surface tension
for i in 1:length(PL)
    if PL.label[i] === :fluid
        PL.position[i] += dx/20 * (rand(3) .- 0.5)
        PL.velocity[i] += [ 0, 0, -impactVelocity ]
    end
end

κ = 4.0 # radius of support multiplier
padFactor = 1.01

NL = constructNeighbourList!(PL, κ, padFactor)

writeBoundaryVTK(PL, saveFolder)

predictorCorrectorTimestepper!(NL, physics, Δtmin, Δtmax, maxiters, maxt, 
    saveΔt, plotΔt, printΔt; saveFolder, onlySaveSummaryData = true, 
    postProcessFunc = droplet_Height_Width_Positions,
    exclLabelLeft=(:fixedGhost,), exclLabelRight=())


##

f = jldopen(joinpath(saveFolder, "savedata.jld2"), "r")

n = f["numFrames"]
times = zeros(n)
heights = zeros(n)
widths = zeros(n)

for i in 0:n-1

    time = f["$i/time"]
    times[i+1] = time

    data = f["$i/data"]
    heights[i+1] = data.height
    widths[i+1] = data.width

end

close(f)

pairwiseForceString = String(pairwiseForceSymbol)

jldsave(joinpath(saveFolder, "dropletSizeData.jld2"); 
    times, heights, widths, R, impactVelocity, dx, s_ff, s_fs, 
    pairwiseForceString, physics)