using ThreadsX, FLoops

function updateBoundaryProperties!(NL::NeighbourList{dim,T}, pressure, physics) where {dim,T}

    # Note: neighbour list not valid for :fixedGhost particles! The neighbour list update
    # skips them intentionally.

    PL = NL.particles
    labels = PL.label
    active = PL.active

    @threads for i in eachParticle(PL)
        if labels[i] === :fluid
            active[i] = true
        elseif labels[i] === :fixedGhost
            active[i] = false
        else
            active[i] = false
        end
    end

    neighbours = NL.neighbours

    @threads for i in eachParticle(PL) # for each particle
        if labels[i] === :fluid # skip non-fluid particles
            for j in neighbours[i] # for each neighbour
                if labels[j] === :fixedGhost # if the neighbour is a fixed ghost
                    active[j] = true # it's active, cause it has a fluid neighbour
                end
            end
        end
    end
    
    Ib = ThreadsX.findall( eachParticle(PL) ) do i 
        @inbounds active[i] && (labels[i] === :fixedGhost)
    end

    if isempty(Ib)
        return nothing
    end

    #= Testing!
    @views begin
        PL.density[Ib] .= physics.ρ₀
        PL.velocity[Ib] .*= 0
        pressure[Ib] .= 0
    end
    return nothing
    =#

    @floop WorkStealingEx() for i in Ib

        @init begin
            A = zeros(4,4)
            a₁ = zeros(4)
            a₂ = zeros(1,4)
            b_ρ = zeros(4)
            b_P = zeros(4)
            ρ_and_∇ρ = zeros(4)
            P_and_∇P = zeros(4)
        end

        interp_i = PL.index[i]

        n = PL.position[i] - PL.position[interp_i]

        PL.density[i] = 0
        PL.velocity[i] *= 0
        pressure[i] = 0

        numFluidNbrs = count(k -> PL.label[k] === :fluid, NL.neighbours[interp_i])
        # sheppardFlag = numFluidNbrs < 5
        sheppardFlag = true

        A .= 0.0
        b_ρ .= 0.0
        b_P .= 0.0
        wsum = 0.0

        @inbounds @fastmath for j in NL.neighbours[interp_i]

            if PL.label[j] !== :fluid
                continue
            end

            wᵢⱼ = Wᵢⱼ(NL, interp_i, j)
            Vⱼ = PL.mass[j] / PL.density[j]
            wsum += wᵢⱼ * Vⱼ

            PL.velocity[i] += PL.velocity[j] * wᵢⱼ * Vⱼ
            PL.density[i] += PL.density[j] * wᵢⱼ * Vⱼ
            # pressure[i] += pressure[j] * wᵢⱼ * Vⱼ

            if !sheppardFlag

                xᵢⱼ = PL.position[interp_i] - PL.position[j]
                
                a₁[1] = wᵢⱼ
                a₁[2:end] .= ∇ᵢWᵢⱼ(NL, interp_i, j)

                a₂[1] = 1.0
                a₂[2:end] .= -xᵢⱼ

                mul!(A, a₁, a₂, Vⱼ, 1.0)

                b_ρ .+= a₁ .* PL.mass[j]
                # b_P .+= pressure[j] .* a₁ .* PL.mass[j] ./ PL.density[j]

            end

        end

        # @assert wsum > 0 "Wsum = $wsum for boundary particle $i"

        if wsum < 1e-2 # neighbourhood deficiency, use fallback values

            PL.velocity[i] *= 0
            PL.density[i] = physics.ρ₀
            # pressure[i] = 0

        elseif sheppardFlag # minor neighbourhood deficiency, use Sheppard

            PL.velocity[i] /= wsum
            PL.density[i] /= wsum
            # pressure[i] /= wsum

        else # big enough neighbourhood, use mDBC

            PL.velocity[i] /= wsum

            # try factorisation
            A_lu = lu!(A, check=false)

            if issuccess(A_lu) # if LU works

                # Get density at boundary by second order correction
                ldiv!(ρ_and_∇ρ, A_lu, b_ρ)
                @views PL.density[i] = ρ_and_∇ρ[1] + (n ⋅ ρ_and_∇ρ[2:end])

                if !(0.98physics.ρ₀ < PL.density[i] < 1.02physics.ρ₀)
                    println("Eek!")
                    @show ρ_and_∇ρ[1]
                    @show ρ_and_∇ρ[2:end]
                    @show cond(A_lu.L * A_lu.U)
                end

                # ldiv!(P_and_∇P, A_lu, b_P)
                # @views pressure[i] = P_and_∇P[1] + (n ⋅ P_and_∇P[2:end])
            
            else # if LU doesn't work

                # Fall back to sheppard interp
                PL.density[i] /= wsum
                # pressure[i] /= wsum

            end

        end

        PL.velocity[i] *= -1.0 # no slip
        
    end

    stateEquation!(pressure, NL, physics, Ib)

    return nothing

end