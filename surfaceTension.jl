@inline function myPolynomial11(r,h)::Float64
    # 1.0 - 23.4959*q^2 + 42.9917*q^3 - 20.4959*q^4
    # p(0) = 1
    # p'(0) = p'(1) = p(1) = 0
    # p(1.1/4) = 0
    if r ≥ h
        return zero(r)
    else
        q = r / h
        # return (1.0 - 23.5*q^2 + 43*q^3 - 20.5*q^4)
        -0.5 * (q - 1)^2 * (-2 - 4q + 41q^2)
    end
end

@inline function myPolynomial16(r, h)::Float64
    # p(0) = 1
    # p'(0) = p'(1) = p(1) = 0
    # p(2/4) = 0
    @fastmath if r >= h
        return zero(r)
    else
        q = r/h
        return (1.0 - 11.0*q^2 + 18.0*q^3 - 8.0*q^4)
    end
end

function pairwiseForceFunc_11_16(r, h, label)
    if label === :fluid
        return myPolynomial11(r,h)
    elseif label === :fixedGhost
        return myPolynomial16(r,h)
    else
        return zero(r)
    end
end

function myPolynomial11b(r,h)
    # p(0) = 1
    # p'(0) = p'(1) = p(1) = 0
    # p(1.5/4) = 0
    @fastmath if r ≥ h
        return zero(r)
    else
        q = r / h
        return 2.25 * (1.0 - 15.4444*q^2 + 26.8889*q^3 - 12.4444*q^4)
    end
end

function pairwiseForceFunc_11b(r, h, label)
    if label === :fluid
        return myPolynomial11b(r,h)
    else
        return zero(r)
    end
end

function pairwiseForceFunc_11b_11b(r, h, label)
    if label === :fluid
        return myPolynomial11b(r,h)
    elseif label === :fixedGhost
        return myPolynomial11b(r,h)
    else
        return zero(r)
    end
end

function zeroVirialPressurePolynomial(r,h)
    # 0.05992 * (420.0 - 3780.0*q^2 + 5880.0*q^3 - 2520.0*q^4)
    # p(0) = 1
    # p'(0) = p'(1) = p(1) = 0
    # p(2.4305/4) = 0
    @fastmath if r ≥ h
        return zero(r)
    else
        q = r / h
        return 0.05992 * (420.0 - 3780.0*q^2 + 5880.0*q^3 - 2520.0*q^4)
    end
end

function pairwiseForceFunc_zeroVirialPressure(r,h,label)
    return zeroVirialPressurePolynomial(r,h)
end





function estimateDropRadius(PL)
    V = sum(eachParticle(PL)) do i
        PL.label[i] === :fluid ? PL.mass[i]/PL.density[i] : 0
    end
    # V = 4/3 pi r^3
    # radius = (3/4 * V / pi)^(1/3)
    dropRadius = cbrt(3/4 * V / pi)
    return dropRadius
end

