using Distances
using Statistics
using StaticArrays
using FLoops, FoldsThreads

export constructNeighbourList!, NeighbourList, ParticleList, updateNeighbourList!,
    resortParticles!, eachParticle, quickParticleList

#############################################################################

struct RegularGrid{d,T}
    origin::SVector{d,T}
    dx::T
    function RegularGrid(origin::Vector{T}, dx::T) where T
        dim = length(origin)
        new{dim,T}( SVector{dim,T}(origin), dx )
    end
end

function Base.show(io::IO, G::RegularGrid)
    println(io, typeof(G))
    println(io, "  Origin = $(G.origin)")
    print(io, "  dx = $(G.dx)")
end

function findGridCell(G::RegularGrid{d,T}, x) where {d,T}
    @inbounds @fastmath cell = ntuple(d) do i
        floor( Int64, (x[i] - G.origin[i]) / G.dx )
    end
    return cell
end

#############################################################################

struct ParticleList{dim,T}
    position::Vector{SVector{dim,T}}
    mass::Vector{T}
    velocity::Vector{SVector{dim,T}}
    density::Vector{T}
    label::Vector{Symbol}
    index::Vector{Int}
    active::Vector{Bool}
    Δx::T
    radius::Vector{T}
    function ParticleList(X, m, V, density, label, index, Δx, radius=zeros(length(X)))
        dim = length(X[1])
        T = eltype(X[1])
        active = fill(true, length(label))
        return new{dim,T}(X, m, V, density, label, index, active, Δx, radius)
    end
end

function quickParticleList(x, Δx, ρ₀, type)

    n = length(x)
    m = fill(Δx^3 * ρ₀, n)
    V = 0 .* x
    density = fill(ρ₀, n)
    label = fill(type, n)
    index = zeros(n)
    radius = fill(Δx/cbrt(4π/3), n)

    return ParticleList(x, m, V, density, label, index, Δx, radius)

end

import Base.*
function *(A::ParticleList, B::ParticleList)
    @assert A.Δx == B.Δx
    position = vcat(A.position, B.position)
    mass = vcat(A.mass, B.mass)
    velocity = vcat(A.velocity, B.velocity)
    density = vcat(A.density, B.density)
    label = vcat(A.label, B.label)
    index = vcat(A.index, B.index)
    Δx = A.Δx
    radius = vcat(A.radius, B.radius)
    return ParticleList(
        position,
        mass,
        velocity,
        density,
        label,
        index,
        Δx,
        radius
    )
end


Base.length(PL::ParticleList) = length(PL.position)
eachParticle(PL::ParticleList) = 1:length(PL)

import Base.permute!
function permute!(PL::ParticleList, p::Vector, ip = invperm(p))
    Base.permute!(PL.position, p)
    Base.permute!(PL.mass, p)
    Base.permute!(PL.velocity, p)
    Base.permute!(PL.density, p)
    Base.permute!(PL.label, p)
    Base.permute!(PL.index, p)
    PL.index[PL.index .!= 0] = ip[PL.index[PL.index .!= 0]]
    Base.permute!(PL.active, p)
    Base.permute!(PL.radius, p)
end

function Base.show(io::IO, PL::ParticleList{dim,T}) where {dim,T}
    println(io, typeof(PL))
end

#############################################################################

struct NeighbourList{dim,T}
    particles::ParticleList{dim,T}
    κ::T # factor in h = κ Δx
    h::T # radius of support of W(r) kernel
    H::T # radius at which to store
    grid::RegularGrid{dim,T}
    cellDict::Dict{ NTuple{dim, Int64}, Vector{UInt64} }
    particle2cell::Vector{ NTuple{dim,Int64} }
    neighbours::Vector{ Vector{UInt64} }
    gridOffsets::Vector{ NTuple{dim,Int64} }
end

Base.length(NL::NeighbourList) = length(NL.particles)
eachParticle(NL::NeighbourList) = eachParticle(NL.particles)

function Base.show(io::IO, NL::NeighbourList)
    println(io, typeof(NL))
    println(io, "  Contains $(length(NL)) particles")
    print(io, "  Radius of support h = $(NL.h)")
end

function constructNeighbourList!(PL::ParticleList{dim,T}, κ, padFactor; 
    exclLabelLeft=(), exclLabelRight=()) where {dim,T}
    
    # In part thanks to "A Parallel SPH Implementation on Multi-Core CPUs", Ihmsen et al.

    h = κ * PL.Δx
    H = padFactor * h

    N = length(PL)

    positions = PL.position # shallow copy!

    grid = RegularGrid(zeros(T,dim), H)
    
    cellDict = buildCellDict(grid, positions, κ)

    activeCells = getActiveCells(cellDict)

    particle2cell = Vector{keytype(cellDict)}(undef, N)
    for cell in activeCells
        for particle in cellDict[cell]
            particle2cell[particle] = cell
        end
    end

    cellContains = maximum( [length(cellDict[c]) for c in activeCells] )
    predictedNbrs = ceil(UInt, 27*cellContains)
    neighbours = [ sizehint!(UInt[], predictedNbrs) for i in 1:N ]

    offsets = getNeighbourCellOffsets(dim)

    NL = NeighbourList{dim,T}(PL, κ, h, H, grid, cellDict, particle2cell, neighbours, offsets)
    computeNeighbours!(NL; exclLabelLeft, exclLabelRight)

    return NL

end

function buildCellDict(grid, positions, κ)
    keyType = typeof(findGridCell(grid, positions[1]))
    cellDict = Dict{keyType, Vector{UInt}}()
    for p in 1:length(positions)
        c = findGridCell(grid, positions[p])
        if c in keys(cellDict)
            push!(cellDict[c], p)
        else
            cellDict[c] = sizehint!([p], ceil(Int, κ^3))
        end
    end
    return cellDict
end

function getActiveCells(cellDict)
    # which cells are used?
    activeCells = findall( !isempty, cellDict )
    return activeCells
end

function spatialSort!(C::Vector{ NTuple{dim,T} }) where {dim,T}
    minInt = minimum(minimum, C)
    f = c -> zindex(c, minInt)
    sort!(C, by=f)
    return nothing
end

function resortParticles!(NL::NeighbourList{dim,T}) where {dim,T}

    N = length(NL)

    activeCells = getActiveCells(NL.cellDict)

    spatialSort!(activeCells)

    perm = sizehint!(Int[], N)
    for cell in activeCells
        append!(perm, NL.cellDict[cell])
    end
    iperm = invperm(perm)

    permute!(NL.particles, perm, iperm)

    for J in values(NL.cellDict)
        J .= iperm[J]
    end

    permute!(NL.neighbours, perm)
    for i in 1:length(NL.neighbours)
        map!(NL.neighbours[i], NL.neighbours[i]) do j
            iperm[j]
        end
    end

    for cell in activeCells
        for i in NL.cellDict[cell]
            NL.particle2cell[i] = cell
        end
    end

    return nothing

end

function getNeighbourCellOffsets(dim)
    offsetTuples = Iterators.product( ntuple(i -> (-1,0,1), dim)... ) |> collect |> vec
end

# function nn(NL::NeighbourList, x, J; exclLabel=Symbol[])
#     jstar = 0
#     dist2 = Inf
#     for j in J
#         if @inbounds(NL.particles.label[j]) in exclLabel
#             continue
#         end
#         dist2j = SqEuclidean(1e-12)(x, @inbounds(NL.particles.position[j]))
#         if dist2j < dist2
#             dist2 = dist2j
#             jstar = j
#         end
#     end
#     return jstar, sqrt(dist2)
# end

# function nn(NL::NeighbourList{dim,T}, x; exclLabel=Symbol[]) where {dim,T}
#     # finds the distance to nearest neighbour in this grid cell or an adjacent one
#     # (it may not exist, in which case return Inf)
#     offsets = NL.gridOffsets
#     cell = findGridCell(NL.grid, x)
#     jstar = 0
#     dist = Inf
#     for off in offsets
#         nbrCell = cell .+ off
#         if nbrCell in keys(NL.cellDict)
#             J = NL.cellDict[nbrCell]
#             j, distj = nn(NL, x, J; exclLabel)
#             if distj < dist
#                 jstar = j
#                 dist = distj
#             end
#         end
#     end
#     return jstar, dist
# end

# function nSearch(NL::NeighbourList, x, J)
#     return filter( j -> SqEuclidean()(x, NL.particles.position[j]) < NL.h^2, J )
# end

# function nSearch(NL::NeighbourList{dim,T}, x; exclLabel=[:boundary]) where {dim,T}
#     offsets = NL.gridOffsets
#     cell = findGridCell(NL.grid, x)
#     neighbours = UInt[]
#     for off in offsets
#         nbrCell = cell .+ off
#         if nbrCell in keys(NL.cellDict)
#             candidates = filter(j -> NL.particles.label[j] ∉ exclLabel, NL.cellDict[nbrCell])
#             append!(neighbours, nSearch(NL,x,candidates))
#         end
#     end
#     return neighbours
# end

function updateNeighbourList!(NL::NeighbourList{dim,T}; exclLabelLeft, exclLabelRight) where {dim,T}

    # In part thanks to "A Parallel SPH Implementation on Multi-Core CPUs", Ihmsen et al.

    moveParticlesCells!(NL)

    computeNeighbours!(NL; exclLabelLeft, exclLabelRight)

end

function moveParticlesCells!(NL)
    
    x = NL.particles.position

    # go through list of moving particles, deleting them from old cells and adding to new
    @inbounds for j in eachParticle(NL)
        newCell = findGridCell(NL.grid, x[j])
        oldCell = NL.particle2cell[j]
        if oldCell == newCell
            continue
        else
            k = findfirst(==(j), NL.cellDict[oldCell])
            deleteat!(NL.cellDict[oldCell], k)
            newCellContents = get!(NL.cellDict, newCell, sizehint!(UInt64[], ceil(Int, NL.κ^3)))
            push!(newCellContents, j)
            NL.particle2cell[j] = newCell
        end
    end

    return nothing

end


function computeNeighbours!(
    NL::NeighbourList{dim,T};
    exclLabelLeft, 
    exclLabelRight
) where {dim,T}


    foreach(empty!, NL.neighbours)
    # @inbounds for i in eachindex(NL.neighbours)
    #     if NL.particles.label[i] ∈ exclLabelLeft
    #         continue
    #     end
    #     NL.neighbours[i] = UInt64[]
    # end

    offsets = NL.gridOffsets

    filter!(p -> !isempty(p.second), NL.cellDict)

    emptyArray = UInt64[]

    # @floop for (cell, cellParticles) in NL.cellDict
    Threads.@threads for cell in collect(keys(NL.cellDict))

        cellParticles = NL.cellDict[cell]

        for off in offsets
            @fastmath nbrCell = cell .+ off
            nbrCellParticles = get(NL.cellDict, nbrCell, emptyArray)
            if isempty(nbrCellParticles)
                continue
            end
            _addNeighbours!(NL, cellParticles, nbrCellParticles, exclLabelLeft, exclLabelRight)
        end

        # for i in cellParticles
        #     if NL.particles.active[i] && (NL.particles.label[i] ∉ exclLabelLeft)
        #         sort!(NL.neighbours[i])
        #     end
        # end

    end
    
    return nothing

end

function _addNeighbours!(
    NL::NeighbourList, I, candidates, exclLabelLeft, exclLabelRight
)

    met = mySqEuc
    H2 = NL.H^2

    neighbours = NL.neighbours
    x = NL.particles.position
    label = NL.particles.label

    @inbounds @fastmath for ki in eachindex(I)

        i = I[ki]

        (label[i] in exclLabelLeft) && continue

        nbrs_i = neighbours[i]
        xi = x[i]
        
        @inbounds @fastmath for kj in eachindex(candidates)

            j = candidates[kj]
            (label[j] in exclLabelRight) && continue

            @inbounds @fastmath if met(xi, x[j]) < H2
                push!(nbrs_i, j)
            end

        end
    end

    return nothing

end

@inline function mySqEuc(x::AbstractArray{T}, y::AbstractArray{T}) where {T}
    d = zero(T)
    for i in eachindex(x)
        @fastmath d += (x[i]-y[i])^2
    end
    return d
    # return mapreduce((a,b) -> (a-b)^2, +, x, y, init=0)
end


function validateNeighbourList(NL; exclLabelLeft, exclLabelRight)

    PL = NL.particles
    x = PL.position
    label = PL.label

    for i in eachParticle(NL)
        nbrs = UInt64[]
        if label[i] ∈ exclLabelLeft
            @info "label[$i] = $(PL.label[i]) doesn't need to know its neighbours" maxlog=100
            continue
        end
        for j in eachParticle(PL)
            if label[j] ∈ exclLabelRight
                @info "label[$j] = $(PL.label[j]) doesn't need to be a neighbour" maxlog=100
                continue
            end
            if SqEuclidean()(x[i], x[j]) < NL.H^2
                push!(nbrs, j)
            end
        end
        givenSet = Set(NL.neighbours[i])
        trueSet = Set(nbrs)
        if givenSet != trueSet
            return i, collect(setdiff(givenSet,trueSet)), collect(setdiff(trueSet,givenSet))
        end
    end

    return 0, nothing, nothing

end