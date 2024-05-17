using LinearAlgebra, Distances

function W(normr,h)
    return phi31(normr, h)
end

function ∇W_scalar(normr,h)
    @fastmath if normr < 1e-10
        return zero(normr)
    else
        return dphi31(normr, h) / (h * normr)
    end
end

function ∇W(r,h)
    return @fastmath ∇W_scalar(norm(r),h) * r
end

function Wᵢⱼ(NL::NeighbourList, i, j)
    @inbounds @fastmath normr = Euclidean()(NL.particles.position[i], NL.particles.position[j])
    return W(normr, NL.h)
end

function ∇ᵢWᵢⱼ(NL::NeighbourList, i, j)
    # What is ∇ᵢ exactly? See Monoghan 1992 "Smoothed Particle Hydrodynamics" p.545
    @inbounds @fastmath r = NL.particles.position[i] - NL.particles.position[j]
    return ∇W(r, NL.h)
end

function ∇ᵢWᵢⱼ_scalar(NL::NeighbourList, i, j)
    @inbounds @fastmath normr = Euclidean()(NL.particles.position[i], NL.particles.position[j])
    return ∇W_scalar(normr, NL.h)
end

function ∇W_fast(xᵢⱼ, rᵢⱼ, h)
    return @fastmath ∇W_scalar(rᵢⱼ, h) * xᵢⱼ
end

#####################################

function cubicSpline(q)
    # base of cubic spline kernels
    if 0 ≤ q ≤ 0.5
        return 6*(q^3 - q^2) + 1
    elseif q ≤ 1
        return 2*(1-q)^3
    else
        return 0.0
    end
end

function dCubicSpline(q)
    if 0 ≤ q ≤ 0.5
        # return 6*(q^3 - q^2) + 1
        return 6*(3*q^2 - 2*q)
    elseif q ≤ 1
        # return 2*(1-q)^3
        return -6*(1-q)^2
    else
        return 0.0
    end
end

function phi31(r, h) # Wendland d = 3, k = 1
     if r < h
        @fastmath q = r/h
        return @fastmath 21 / (2 * pi * h^3) * (1 - q)^4 * (4*q + 1)
    else
        return zero(r)
    end
end

function dphi31(r, h)
     if r < h
        @fastmath q = r/h
        return @fastmath 21 / (2 * pi * h^3) * -20 * (1 - q)^3 * q
    else
        return zero(r)
    end
end

function phi42(r, h) # Wendland d = 3, k = 2
    if r < h
        @fastmath q = r/h
        return @fastmath 165/(32*pi*h^3) * (1 - q)^6 * (35*q^2 + 18*q + 3)
    else
        return zero(q)
    end
end

function dphi42(r, h)
    if r < h
        @fastmath q = r/h
        # return @fastmath 165/(32*pi*h^3) * 
        #     ( -6*(1-q)^5 * (35*q^2 + 18*q + 3) +
        #     (1 - q)^6 * (70*q^2 + 18) )
        return @fastmath 165/(32*pi*h^3) * -56 * q * (5*q + 1) * (1 - q)^5
    else
        return zero(r)
    end
end

function C(r,h) # kernel for surface tension by Akinci 2013
    nr = norm(r)
    if h/2 < nr ≤ h
        @fastmath result = 32/pi/h^9 * (h-nr)^3 * nr^3
    elseif 0 < nr ≤ h/2
        @fastmath result = 32/pi/h^9 * (2 * (h-nr)^3 * nr^3 - h^6/64)
    else
        return zero(r)
    end
    return result * r / nr
end

function Cᵢⱼ(NL::NeighbourList, i, j)
    return C(@inbounds(NL.particles.position[i] - NL.particles.position[j]), NL.h )
end

function A(r,h) # kernel for adhesion by Akinci 2013
    nr = norm(r)
    if h/2 < nr ≤ h
        return @fastmath 0.007/h^(3.25) * (-4nr^2/h + 6nr - 2*h)^(1/4) / nr * r
    else
        return zero(r)
    end
end