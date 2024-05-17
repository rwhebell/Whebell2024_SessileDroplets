using LinearAlgebra, LoopVectorization, ProgressBars, Printf, Distances
using Base.Threads: @threads, threadid, nthreads
using FLoops


function stateEquation!(pressure, NL, physics, If)

    ρ = NL.particles.density
    c₀ = physics.c₀
    ρ₀ = physics.ρ₀

    # map!(view(pressure, If), view(ρ, If)) do ρᵢ
    #     c₀^2 * ρ₀ / 7 * ( (ρᵢ/ρ₀)^7 - 1 )
    #     # c₀^2 * ( ρᵢ - ρ₀ )
    # end

    @threads for i in If
        @fastmath @inbounds pressure[i] = c₀^2 * ρ₀ / 7 * ( (ρ[i]/ρ₀)^7 - 1 )
    end

    return nothing

end


function calcInteractions!(DρDt, DρDt_diffusion, DvDt, DvDt_viscous, NL::NeighbourList{dim,T}, P, physics, If) where {dim,T}

    PL = NL.particles
    m = PL.mass
    ρ = PL.density
    x = PL.position
    v = PL.velocity
    label = PL.label

    α = physics.α
    μ = physics.μ
    c₀ = physics.c₀
    h = NL.h
    ρ₀ = physics.ρ₀
    g = physics.g
    δ = physics.δ

    s = physics.s
    F = physics.F

    @threads for i in If
       
        DρDt[i] *= 0
        DρDt_diffusion[i] *= 0

        DvDt[i] *= 0
        DvDt_viscous[i] *= 0

        # gravity
        DvDt[i] += g

        @fastmath @inbounds for j in NL.neighbours[i]

            if (i != j)

                xᵢⱼ = x[i] - x[j]
                vᵢⱼ = v[i] - v[j]
                rᵢⱼ = norm(xᵢⱼ)
                Vⱼ = m[j]/ρ[j]
                gradW = ∇W_fast(xᵢⱼ, rᵢⱼ, h)

                # ρ div v
                DρDt[i] += ρ[i] * (vᵢⱼ ⋅ gradW) * Vⱼ

                # delta-SPH per [Molteni and Colagrossi, 2009] and [Antuono et al 2010]
                # http://dx.doi.org/10.1016/j.cpc.2009.11.002
                if δ != 0
                    ψᵢⱼ = 2 * (ρ[i] - ρ[j]) / (rᵢⱼ^2 + 0.01h^2)
                    DρDt_diffusion[i] += δ * h * c₀ * ψᵢⱼ * (xᵢⱼ ⋅ gradW) * Vⱼ
                end

                # grad P
                # Bonet & Lok (1999)
                DvDt[i] += (-(P[i] + P[j]) * Vⱼ / ρ[i]) * gradW

                # viscosity
                Π = ( vᵢⱼ ⋅ xᵢⱼ ) / (rᵢⱼ^2)
                if physics.viscous
                    # Colagrossi et al 2011 says this Monaghan & Gingold formula is better 
                    # than Morris formula for free surface flows
                    # https://doi.org/10.1103/PhysRevE.84.026705
                    K = 10 # 2*(dim + 2)
                    DvDt_viscous[i] += (μ * K / ρ[i]) * Π * Vⱼ * gradW
                else
                    # https://doi.org/10.1016/j.cpc.2014.10.004, p871
                    if Π < 0
                        DvDt_viscous[i] += (α * c₀ * h * ρ[j] * 2/(ρ[i]+ρ[j])) * Π * Vⱼ * gradW
                    end
                end

                # interparticle forces for interfacial tension
                labelⱼ = label[j]
                sᵢⱼ = get(s, labelⱼ, 0)
                if sᵢⱼ != 0
                    Fᵢⱼ = F(rᵢⱼ, h, labelⱼ)
                    DvDt[i] += (sᵢⱼ * h * Fᵢⱼ / rᵢⱼ / m[i]) * xᵢⱼ
                end

            end

        end

    end

    return nothing

end


function calcTimestep_predictorCorrector(NL, physics, DvDt, If)

    PL = NL.particles
    dtv = physics.viscous ? PL.Δx^2 * physics.ρ₀ / physics.μ : Inf
    max_acc = maximum(norm, @view(DvDt[If]))
    dta = 0.15 * sqrt( PL.Δx / max_acc )
    dtc = 0.15 * PL.Δx / physics.c₀

    return min(dtv, dta, dtc)

end


function predictorCorrectorTimestepper!(
    NL::NeighbourList{dim,T}, physics::PhysicalParameters{dim,T}, 
    Δtmin, Δtmax, maxiters, maxt, saveΔt, plotΔt, printΔt; saveFolder="",
    postProcessFunc = p -> missing, onlySaveSummaryData = false,
    exclLabelLeft = (:fixedGhost,), exclLabelRight = ()
) where {dim,T}

    doSave = !isempty(saveFolder) && !isinf(saveΔt)
    doPlot = !isempty(saveFolder) && !isinf(plotΔt)
    doPrint = !isinf(printΔt)

    includeBoundary = false
    
    time = zero(T)
    nextSaveTime = saveΔt
    nextPlotTime = plotΔt
    nextPrintTime = printΔt

    PL = NL.particles

    If = findall(isequal(:fluid), PL.label)

    DρDt = 0 .* PL.density
    DρDt_diffusion = 0 .* PL.density

    DvDt = 0 .* PL.velocity
    DvDt_diffusion = 0 .* PL.velocity

    oldρ = 0 .* PL.density[If]
    oldv = 0 .* PL.velocity[If]
    oldx = 0 .* PL.position[If]

    pressure = 0 * PL.density

    updateNeighbourList!(NL; exclLabelLeft, exclLabelRight)
    stateEquation!(pressure, NL, physics, If)

    if doSave
        mkpath(saveFolder)
        V1 = JLD2FrameWriter(saveFolder)
        writeSnapshot(V1, PL, time; f=postProcessFunc, onlySaveSummaryData)
    end

    if doPlot
        mkpath(saveFolder)
        V2 = VTKWriter(saveFolder)
        writeSnapshot(V2, PL, pressure, time; includeBoundary)
    end

    if doSave || doPlot
        writeSettingsLog(saveFolder * "/settings.log", NL, physics, plotΔt)
    end
    
    it = 0

    c₀est = 0.0

    while it < maxiters && time ≤ maxt

        # Get adaptive timestep
        if it == 0
            Δt = Δtmin
        else
            Δt = min( Δtmax, max(Δtmin, calcTimestep_predictorCorrector(NL, physics, DvDt, If)) )
        end

        # Occassionally resort particles (does NOT resort things like DvDt or pressure)
        if it % 10_000 == 0
            resortParticles!(NL)
            If = findall(isequal(:fluid), PL.label)
            pressure .= 0
        end

        # Update neighbours, apply equation of state
        updateNeighbourList!(NL; exclLabelLeft, exclLabelRight)
        stateEquation!(pressure, NL, physics, If)

        # ------------------ <Printing and saving> ------------------
        pmax = maximum(view(pressure, If))
        sqrtpmaxonrho0 = sqrt(max(pmax,0)/physics.ρ₀)
        umax = maximum(norm, view(PL.velocity, If))
        c₀est = max( 10*max(umax, sqrtpmaxonrho0), c₀est )
        maxρerr = maximum(If) do i
            abs(PL.density[i] - physics.ρ₀) / physics.ρ₀
        end
        maxpres = maximum(view(pressure, If))

        if doSave && time ≥ nextSaveTime
            nextSaveTime += saveΔt
            writeSnapshot(V1, PL, time; f = postProcessFunc, onlySaveSummaryData)
        end

        if doPlot && time ≥ nextPlotTime
            nextPlotTime += plotΔt
            writeSnapshot(V2, PL, pressure, time; includeBoundary)
        end
        
        if doPrint && time ≥ nextPrintTime
            nextPrintTime += printΔt
            println(@sprintf("t = %.5f, dt = %.1e, c₀est = %.1e, max|ρ-ρ₀|/ρ₀ = %.1e, maxp = %.1e", time, Δt, c₀est, maxρerr, maxpres))
            if c₀est > physics.c₀
                @debug @sprintf("c₀ = %.2e might be too small at t=%.5f", c₀est, time)
            end
            if maxρerr > 0.01
                @warn @sprintf("max ρ error = %.2e (above 1%%) at t=%.5f", maxρerr, time)
            end
            c₀est = 0.0
        end
        # ----------------- </Printing and saving> ------------------

        ## Copy previous timestep to storage
        oldρ .= @view PL.density[If]
        oldv .= @view PL.velocity[If]
        oldx .= @view PL.position[If]

        ## Calculate DvDt and DρDt at timestep n
        for i in If
            DρDt[i] *= 0
            DρDt_diffusion[i] *= 0
            DvDt[i] *= 0
            DvDt_diffusion[i] *= 0
        end

        calcInteractions!(DρDt, DρDt_diffusion, DvDt, DvDt_diffusion, NL, pressure, physics, If)
        
        ## Perform a half-step to timestep n + 1/2
        @fastmath @inbounds for (k,i) in enumerate(If)
            PL.velocity[i] = oldv[k] + Δt/2 * (DvDt[i] + DvDt_diffusion[i])
            PL.position[i] = oldx[k] + Δt/2 * oldv[k]
            PL.density[i] = oldρ[k] + Δt/2 * (DρDt[i] + DρDt_diffusion[i])
        end

        ## Update stuff
        # updateNeighbourList!(NL; exclLabelLeft, exclLabelRight) # skip for speed
        stateEquation!(pressure, NL, physics, If)

        ## Calculate DvDt and DρDt at timestep n + 1/2
        for i in If
            DρDt[i] *= 0
            DρDt_diffusion[i] *= 0
            DvDt[i] *= 0
            DvDt_diffusion[i] *= 0
        end

        calcInteractions!(DρDt, DρDt_diffusion, DvDt, DvDt_diffusion, NL, pressure, physics, If)

        ## Perform a full step to timestep n + 1
        @fastmath @inbounds for (k,i) in enumerate(If)
            PL.velocity[i] = oldv[k] + Δt * (DvDt[i] + DvDt_diffusion[i])
            PL.position[i] = oldx[k] + Δt/2 * (oldv[k] + PL.velocity[i])
            PL.density[i] = oldρ[k] + Δt * (DρDt[i] + DρDt_diffusion[i])
        end

        time += Δt
        it += 1

    end

    if doSave
        writeSnapshot(V1, PL, time; f = postProcessFunc)
        close(V1)
    end

    if doPlot
        writeSnapshot(V2, PL, pressure, time; includeBoundary)
        close(V2)
    end

    if doPrint
        println("$it iters taken")
    end

end