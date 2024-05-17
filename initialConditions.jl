using StaticArrays, LinearAlgebra, Statistics

function gridPoints(u,v,d)
    dim = length(u)
    @assert length(v) == dim
    if length(d) == 1
        D = [d for k in 1:dim]
    end
    ranges = [ range(u[i], v[i], step=D[i]) for i in 1:dim ]
    return SVector{dim}.(Iterators.product(ranges...) |> collect |> vec)
end

function twommDrop_fgp(ρ₀, dxf=8e-5, dxb=dxf)

    a = 1e-3
    Xf = Iterators.product(-a:dxf:a, -a:dxf:a, -a:dxf:a) |> collect |> vec
    Xf = [ SVector{3}(Xf[j]...) for j in 1:length(Xf) ]
    filter!( x -> norm(x) ≤ a, Xf )
    Xf_c = mean(Xf)
    Xf .-= [Xf_c]

    # Boundary / fixed ghost particles
    bound_z = -2e-3
    xy_extent = 3a
    
    Xb = Iterators.product(
        -xy_extent:dxb:xy_extent, 
        -xy_extent:dxb:xy_extent, 
        bound_z .- dxb.*(0.5:1.0:4.5)
    ) |> collect |> vec

    Xb = [ SVector{3}(Xb[j]...) for j in 1:length(Xb) ]
    
    # Interpolation points for ghost particles
    Xinterp = (Xb .- ([0,0,bound_z],))
    for i in axes(Xinterp,1)
        Xinterp[i] = Xinterp[i] .* [1,1,-1] + [0,0,bound_z]
    end

    X = [Xb; Xinterp; Xf]

    Nb = length(Xb)
    Nf = length(Xf)
    N = Nf + 2Nb
    println("Using $Nf (fluid) + $Nb (boundary) particles")

    mf_unit = dxf^3 * ρ₀
    mb_unit = dxb^3 * ρ₀
    m = [ mb_unit .* ones(Nb); zeros(Nb); mf_unit .* ones(Nf) ]

    V = 0 .* X

    density = [fill(ρ₀, Nb); fill(0, Nb); fill(ρ₀, Nf)]

    radius = (3/4/π .* m./density).^(1/3)
    radius[isnan.(radius)] .= 0

    label = [fill(:fixedGhost, Nb); fill(:interpPoint, Nb); fill(:fluid, Nf)]

    index = Int64[ Nb+1:2Nb; zeros(Nb); zeros(Nf) ]

    PL = ParticleList(X, m, V, density, label, index, dxf, radius)

    return PL, dxf

end

function onemmDrop_fgp(ρ₀, dxf, dxb=dxf)

    a = 0.5e-3
    Xf = Iterators.product(-a:dxf:a, -a:dxf:a, -a:dxf:a) |> collect |> vec
    Xf = [ SVector{3}(Xf[j]...) for j in 1:length(Xf) ]
    filter!( x -> norm(x) ≤ a, Xf )
    Xf .+= [[0, 0, a + 2dxf]]

    # Boundary / fixed ghost particles
    bound_z = 0.0
    xy_extent = 5a
    
    Xb = Iterators.product(
        -xy_extent:dxb:xy_extent, 
        -xy_extent:dxb:xy_extent, 
        bound_z .- dxb.*(0.5:1.0:4.5)
    ) |> collect |> vec

    Xb = [ SVector{3}(Xb[j]...) for j in 1:length(Xb) ]
    
    # Interpolation points for ghost particles
    Xinterp = (Xb .- ([0,0,bound_z],))
    for i in axes(Xinterp,1)
        Xinterp[i] = Xinterp[i] .* [1,1,-1] + [0,0,bound_z]
    end

    X = [Xb; Xinterp; Xf]

    Nb = length(Xb)
    Nf = length(Xf)
    N = Nf + 2Nb
    println("Using $Nf (fluid) + $Nb (boundary) particles")

    mf_unit = dxf^3 * ρ₀
    mb_unit = dxb^3 * ρ₀
    m = [ mb_unit .* ones(Nb); zeros(Nb); mf_unit .* ones(Nf) ]

    V = 0 .* X

    density = [fill(ρ₀, Nb); fill(0, Nb); fill(ρ₀, Nf)]

    radius = (3/4/π .* m./density).^(1/3)
    radius[isnan.(radius)] .= 0

    label = [fill(:fixedGhost, Nb); fill(:interpPoint, Nb); fill(:fluid, Nf)]

    index = Int64[ Nb+1:2Nb; zeros(Nb); zeros(Nf) ]

    PL = ParticleList(X, m, V, density, label, index, dxf, radius)

    return PL

end

function r_mm_drop(ρ₀, r, dxf; dxb=dxf, xy_extent=3r)

    foo = [ dxf/2:dxf:r; -dxf/2:-dxf:-r ]
    Xf = [ SVector{3,Float64}(x1, x2, x3) 
        for x1 in foo, x2 in foo, x3 in foo if norm([x1,x2,x3]) < r ]
    Xf .+= [[0, 0, r + 2dx]]
    
    X1 = X2 = [ dxb/2:dxb:xy_extent; -dxb/2:-dxb:-xy_extent ]
    X3 = -dxb/2:-dxb:-5dxb
    Xb = [ SVector{3,Float64}(x1,x2,x3) for x1 in X1, x2 in X2, x3 in X3 ] |> vec

    X = [Xb; Xf]

    Nb = length(Xb)
    Nf = length(Xf)
    N = Nf + Nb
    println("Using $Nf (fluid) + $Nb (boundary) particles")

    mf_unit = dxf^3 * ρ₀
    mb_unit = dxb^3 * ρ₀
    m = [ mb_unit .* ones(Nb); mf_unit .* ones(Nf) ]

    V = 0 .* X

    density = [fill(ρ₀, Nb); fill(ρ₀, Nf)]

    radius = cbrt.(3/4/π .* m./density)
    radius[isnan.(radius)] .= 0

    label = [fill(:fixedGhost, Nb); fill(:fluid, Nf)]

    index = zeros(Int64, N)

    PL = ParticleList(X, m, V, density, label, index, dxf, radius)

    return PL

end


function r_mm_hemi_drop_CAcal(ρ₀, r, dxf)

    dxb = dxf
    
    # Boundary / fixed ghost particles
    bound_z = 0.0
    xy_extent = 3r
    
    Xf = Iterators.product(-r:dxf:r, -r:dxf:r, 0:dxf:r) |> collect |> vec
    Xf = [ SVector{3}(Xf[j]...) for j in 1:length(Xf) ]
    filter!( x -> norm(x) ≤ r, Xf )
    Xf .+= [[0, 0, dxf/2]]

    Xb = Iterators.product(
        -xy_extent:dxb:xy_extent, 
        -xy_extent:dxb:xy_extent, 
        bound_z .- dxb.*(0.5:1.0:4.5)
    ) |> collect |> vec

    Xb = [ SVector{3}(Xb[j]...) for j in 1:length(Xb) ]

    X = [Xb; Xf]

    Nb = length(Xb)
    Nf = length(Xf)
    N = Nf + Nb
    println("Using $Nf (fluid) + $Nb (boundary) particles")

    mf_unit = dxf^3 * ρ₀
    mb_unit = dxb^3 * ρ₀
    m = [ mb_unit .* ones(Nb); mf_unit .* ones(Nf) ]

    V = 0 .* X

    density = [fill(ρ₀, Nb); fill(ρ₀, Nf)]

    radius = cbrt.(3/4/π .* m./density)
    radius[isnan.(radius)] .= 0

    label = [fill(:fixedGhost, Nb); fill(:fluid, Nf)]

    index = zeros(Int64, N)

    PL = ParticleList(X, m, V, density, label, index, dxf, radius)

    return PL

end


function onemmDrop(ρ₀, dxf, dxb=dxf)

    a = 0.5e-3
    Xf = Iterators.product(-a:dxf:a, -a:dxf:a, -a:dxf:a) |> collect |> vec
    Xf = [ SVector{3}(Xf[j]...) for j in 1:length(Xf) ]
    filter!( x -> norm(x) ≤ a, Xf )
    Xf .+= [[0, 0, a + 2dxf]]

    # Boundary / fixed ghost particles
    bound_z = 0.0
    xy_extent = 5a
    
    Xb = Iterators.product(
        -xy_extent:dxb:xy_extent, 
        -xy_extent:dxb:xy_extent,
        bound_z .- dxb.*(0.5:1.0:2.5)
    ) |> collect |> vec

    Xb = [ SVector{3}(Xb[j]...) for j in 1:length(Xb) ]

    X = [Xb; Xf]

    Nb = length(Xb)
    Nf = length(Xf)
    N = Nf + Nb
    println("Using $Nf (fluid) + $Nb (boundary) particles")

    mf_unit = dxf^3 * ρ₀
    mb_unit = dxb^3 * ρ₀
    m = [ mb_unit .* ones(Nb); mf_unit .* ones(Nf) ]

    V = 0 .* X

    density = [fill(ρ₀, Nb); fill(ρ₀, Nf)]

    radius = (3/4/π .* m./density).^(1/3)
    radius[isnan.(radius)] .= 0

    label = [fill(:fixedGhost, Nb); fill(:fluid, Nf)]

    index = zeros(Int64, N)

    PL = ParticleList(X, m, V, density, label, index, dxf, radius)

    return PL

end

function twommDrop(ρ₀, dxf=8e-5, dxb=dxf)

    a = 1e-3
    Xf = Iterators.product(-a:dxf:a, -a:dxf:a, -a:dxf:a) |> collect |> vec
    Xf = [ SVector{3}(Xf[j]...) for j in 1:length(Xf) ]
    filter!( x -> norm(x) ≤ a, Xf )
    Xf_c = mean(Xf)
    Xf .-= [Xf_c]

    # Boundary / fixed ghost particles
    bound_z = -2e-3
    xy_extent = 3a
    
    Xb = Iterators.product(
        -xy_extent:dxb:xy_extent, 
        -xy_extent:dxb:xy_extent, 
        bound_z .- dxb.*(0.5:1.0:4.5)
    ) |> collect |> vec

    Xb = [ SVector{3}(Xb[j]...) for j in 1:length(Xb) ]
    
    X = [Xb; Xf]

    Nb = length(Xb)
    Nf = length(Xf)
    N = Nf + Nb
    println("Using $Nf (fluid) + $Nb (boundary) particles")

    mf_unit = dxf^3 * ρ₀
    mb_unit = dxb^3 * ρ₀
    m = [ mb_unit .* ones(Nb); mf_unit .* ones(Nf) ]

    V = 0 .* X

    density = [fill(ρ₀, Nb); fill(ρ₀, Nf)]

    radius = (3/4/π .* m./density).^(1/3)
    radius[isnan.(radius)] .= 0

    label = [fill(:fixedGhost, Nb); fill(:fluid, Nf)]

    index = Int64[ zeros(Nb); zeros(Nf) ]

    PL = ParticleList(X, m, V, density, label, index, dxf, radius)

    return PL, dxf

end

function tank_fgp(ρ₀, dxf)

    dx = dxf
    h = 3*dxf
    dim = 3

    # Wall locations: x,y,z
    a = 1.0
    b = 1.0
    c = 1.0

    # Edge of support radius: x,y,z
    A = a + h
    B = b + h
    C = c + h

    x = gridPoints([a, 0, 0] .+ dx/2, [A, b, c] .- dx/2, dx)
    xint = SVector{dim}[ [2a - x[i][1], x[i][2], x[i][3]] for i in 1:length(x) ]

    y = gridPoints([0, b, 0] .+ dx/2, [a, B, c] .- dx/2, dx)
    yint = SVector{dim}[ [y[i][1], 2b - y[i][2], y[i][3]] for i in 1:length(y) ]

    z = gridPoints([0, 0, c] .+ dx/2, [a, b, C] .- dx/2, dx)
    zint = SVector{dim}[ [z[i][1], z[i][2], 2c - z[i][3]] for i in 1:length(z) ]

    xy = gridPoints([a, b, 0] .+ dx/2, [A, B, c] .- dx/2, dx)
    xyint = SVector{dim}[ [2a - xy[i][1], 2b - xy[i][2], xy[i][3]] for i in 1:length(xy) ]

    xz = gridPoints([a, 0, c] .+ dx/2, [A, b, C] .- dx/2, dx)
    xzint = SVector{dim}[ [2a - xz[i][1], xz[i][2], 2c - xz[i][3]] for i in 1:length(xz) ]

    yz = gridPoints([0, b, c] .+ dx/2, [a, B, C] .- dx/2, dx)
    yzint = SVector{dim}[ [yz[i][1], 2b - yz[i][2], 2c - yz[i][3]] for i in 1:length(yz) ]

    xyz = gridPoints([a, b, c] .+ dx/2, [A, B, C] .- dx/2, dx)
    xyzint = SVector{dim}[ (2*[a,b,c] - xyz[i]) for i in 1:length(xyz) ]

    P_boundary = cat(x, y, z, xy, xz, yz, xyz, dims=1)
    P_interp = cat(xint, yint, zint, xyint, xzint, yzint, xyzint, dims=1)

    # x reflection
    append!(P_boundary, ([-1 0 0; 0 1 0 ; 0 0 1],) .* P_boundary)
    append!(P_interp, ([-1 0 0; 0 1 0 ; 0 0 1],) .* P_interp)

    # y reflection
    append!(P_boundary, ([1 0 0; 0 -1 0 ; 0 0 1],) .* P_boundary)
    append!(P_interp, ([1 0 0; 0 -1 0 ; 0 0 1],) .* P_interp)

    # z reflection
    append!(P_boundary, ([1 0 0; 0 1 0 ; 0 0 -1],) .* P_boundary)
    append!(P_interp, ([1 0 0; 0 1 0 ; 0 0 -1],) .* P_interp)

    P_fluid = gridPoints([-a,-b,-c] .+ dxf/2, [a,b,0] .- dxf/2, dxf)

    X = [ P_fluid; P_boundary; P_interp ]

    Nb = length(P_boundary)
    Nf = length(P_fluid)
    println("Using $Nf (fluid) + $Nb (boundary) particles")

    N = length(X)

    mass = [ fill( dxf^3 * ρ₀, Nf ); fill( dx^3 * ρ₀, Nb ); zeros(length(P_interp)) ]

    V = [ @SVector zeros(3) for i in 1:N ]

    density = ρ₀ .* ones(N)

    label = [ fill(:fluid, length(P_fluid)); 
        fill(:fixedGhost, length(P_boundary));
        fill(:interpPoint, length(P_interp)) ]

    index = zeros(N)

    mass[label .=== :interpPoint] .= 0
    density[label .=== :interpPoint] .= 0
    index[label .=== :fixedGhost] = ((length(P_fluid)+length(P_boundary))+1):N

    PL = ParticleList(X, mass, V, density, label, index)

    return PL, dx

end

function zeroGravDroplet(ρ₀, dx, r)

    X = Iterators.product(-r:dx:r, -r:dx:r, -r:dx:r) |> collect |> vec
    X = [ SVector{3}(X[j]...) for j in 1:length(X) ]
    filter!( x -> norm(x) <= r, X )

    N = length(X)

    m_unit = dx^3 * ρ₀
    m = m_unit .* ones(N)

    V = 0 .* X

    density = fill(ρ₀, N)

    r = dx / ∛(4pi/3)
    radius = fill(r, N)

    label = fill(:fluid, N)

    index = zeros(Int64, N)

    PL = ParticleList(X, m, V, density, label, index, dx, radius)

    return PL

end

function sineSurface(ρ₀, dx, a, T, r, xy_extent)

    # Boundary particles
    dxb = dx

    f(x) = x[3] - a*sin(2π*x[1]/T)
    df(x) = [-2π/T * a*cos(2π*x[1]/T), 0, 1.0]
    sdf(x) = f(x) / norm(df(x))
    dist(x) = abs(sdf(x))

    Xb = Iterators.product(
        -xy_extent:dxb:xy_extent, 
        -xy_extent:dxb:xy_extent, 
        a:(-dxb):-4dxb
    ) |> collect |> vec
    Xb = [ SVector{3}(Xb[j]...) for j in 1:length(Xb) ]

    filter!(Xb) do xb
        sdfxb = sdf(xb)
        abs(sdfxb) ≤ 4*dx && sdfxb ≤ 0
    end

    # Interpolation points
    Xi = [ xb + 2dist(xb) * normalize(df(xb)) for xb in Xb ]

    # Fluid particles
    Xf = Iterators.product(-r:dx:r, -r:dx:r, -r:dx:r) |> collect |> vec
    Xf = [ SVector{3}(Xf[j]...) for j in 1:length(Xf) ]
    filter!( x -> norm(x) <= r, Xf )
    Xf .-= (mean(Xf),)

    newDropCentre = a + r + 3dx
    Xf .+= ([0, 0, newDropCentre],)

    # Other properties
    Nf = length(Xf)
    Nb = length(Xb)
    Ni = Nb
    N = Nf + Nb + Ni

    X = [ Xi; Xb; Xf ]
    m = [ zeros(Ni); (ρ₀ * dx^3)*ones(Nb+Nf) ]
    v = 0 .* X
    ρ = [ zeros(Ni); ρ₀*ones(Nb+Nf) ]
    label = [ fill(:interpPoint,Ni); fill(:fixedGhost,Nb); fill(:fluid,Nf) ]
    index = [ zeros(Ni); collect(1:Nb); zeros(Nf) ]
    Δx = dx
    radius = ∛(3/4/pi * dx^3) * ones(N)

    return ParticleList(X, m, v, ρ, label, index, Δx, radius)

end


function initSPHfromImplicit(ρ₀, dx, f, ∇f, r, mins, maxs)

    # ASSUMES NEGATIVE f IS INSIDE BOUNDARY!!

    sdf(x) = f(x) / norm(∇f(x))
    dist(x) = abs(sdf(x))

    # Boundary particles
    dxb = dx

    Xb = Iterators.product(
        mins[1]:dxb:maxs[1], 
        mins[2]:dxb:maxs[2], 
        mins[3]:dxb:maxs[3]
    ) |> collect |> vec
    Xb = [ SVector{3}(Xb[j]...) for j in 1:length(Xb) ]

    filter!(Xb) do xb
        sdfxb = sdf(xb)
        abs(sdfxb) ≤ 4*dx && sdfxb < 0 # must be strict < to avoid interp pt == boundary pt
    end

    # Interpolation points
    Xi = [ xb + 2dist(xb) * normalize(∇f(xb)) for xb in Xb ]

    # Fluid particles
    Xf = Iterators.product(-r:dx:r, -r:dx:r, -r:dx:r) |> collect |> vec
    Xf = [ SVector{3}(Xf[j]...) for j in 1:length(Xf) ]
    filter!( x -> norm(x) <= r, Xf )
    Xf .-= (mean(Xf),)

    newDropCentre = maximum(x -> x[3], Xb) + r + 3dx
    Xf .+= ([0, 0, newDropCentre],)

    # Other properties
    Nf = length(Xf)
    Nb = length(Xb)
    Ni = Nb
    N = Nf + Nb + Ni

    X = [ Xi; Xb; Xf ]
    m = [ zeros(Ni); (ρ₀ * dx^3)*ones(Nb+Nf) ]
    v = 0 .* X
    ρ = [ zeros(Ni); ρ₀*ones(Nb+Nf) ]
    label = [ fill(:interpPoint,Ni); fill(:fixedGhost,Nb); fill(:fluid,Nf) ]
    index = [ zeros(Ni); collect(1:Nb); zeros(Nf) ]
    Δx = dx
    radius = ∛(3/4/pi * dx^3) * ones(N)

    return ParticleList(X, m, v, ρ, label, index, Δx, radius)

end

insideEllipse(x,r) = mapreduce( (xi,ri) -> (xi/ri)^2, +, x, r ) < 1

function ellipticalDroplet(ρ₀, dx, r)

    X = Iterators.product( ntuple(i -> -r[i]:dx:r[i], length(r))... ) |> collect |> vec
    X = [ SVector{3}(X[j]...) for j in 1:length(X) ]
    filter!( x -> insideEllipse(x,r), X )
    X .-= (mean(X),)

    N = length(X)

    m_unit = dx^3 * ρ₀
    m = m_unit .* ones(N)

    V = 0 .* X

    density = fill(ρ₀, N)

    particle_radius = fill( dx / ∛(4pi/3), N )

    label = fill(:fluid, N)

    index = zeros(Int64, N)

    PL = ParticleList(X, m, V, density, label, index, dx, particle_radius)

    return PL

end

function fluidSphere(dx, R, ρ₀)
    foo = [ dx/2:dx:R; -dx/2:-dx:-R ]
    x = [ SVector{3,Float64}(x1, x2, x3) for x1 in foo, x2 in foo, x3 in foo if norm([x1,x2,x3]) < R ]
    return quickParticleList(x, dx, ρ₀, :fluid)
end

function pillaredSurface(dx, L, d, w, height, ρ₀, label)

    H(t) = t >= 0 ? one(t) : zero(t)
    boxcar(t, T, w) = H(t - T + w/2) - H(t - T - w/2)

    # pillar locations
    Y1 = Y2 = [ 0:d:L; -d:-d:-L ] .+ dx/2
    np = length(Y1)

    pillar(x1, x2) = sum(i -> boxcar(x1, Y1[i], w), 1:np) * sum(j -> boxcar(x2, Y2[j], w), 1:np)

    func(x1,x2,x3) = x3 < (height * (pillar(x1,x2)-1))

    bottom = -height - 5dx
    X1 = X2 = [0:dx:L; -dx:-dx:-L]
    X3 = 0:-dx:bottom
    x = [ SVector{3,Float64}(x1-dx/2, x2-dx/2, x3) for x1 in X1, x2 in X2, x3 in X3 if func(x1,x2,x3) ]

    return quickParticleList(x, dx, ρ₀, label)

end

function leftRightChemicallyPatterned(dx, L, ρ₀)

    X1 = X2 = [ dx/2:dx:L; -dx/2:-dx:-L ]
    X3 = -dx/2:-dx:-5dx
    x = [ SVector{3,Float64}(x1,x2,x3) for x1 in X1, x2 in X2, x3 in X3 ] |> vec

    PL = quickParticleList(x, dx, ρ₀, :dummy)

    for i in eachParticle(PL)
        if PL.position[i][2] < 0
            PL.label[i] = :left
        else
            PL.label[i] = :right
        end
    end

    return PL

end

function stripedChemically(dx, L, ρ₀, T)

    X1 = X2 = [ dx/2:dx:L; -dx/2:-dx:-L ]
    X3 = -dx/2:-dx:-5dx
    x = [ SVector{3,Float64}(x1,x2,x3) for x1 in X1, x2 in X2, x3 in X3 ] |> vec

    PL = quickParticleList(x, dx, ρ₀, :dummy)

    for i in eachParticle(PL)
        if mod(PL.position[i][2], 2T) < T
            PL.label[i] = :A
        else
            PL.label[i] = :B
        end
    end

    return PL

end