using FileIO, JLD2, CodecZlib
import Base.close 
using Printf
using LightXML
using WriteVTK


function writeFrameOBJ(PL::ParticleList, frameNum, folder; scale=1, digits=0)

    if frameNum == 0 && !isdir("./" * folder)
        mkdir("./" * folder)
    end

    X = PL.position
    fout = open("./$folder/frame.$frameNum.obj", "w")
    for i = 1:length(X)
        if PL.label[i] === :fluid || PL.label[i] === :fixedGhost
            println(fout, @sprintf("v %f %f %f", X[i][1]/scale, X[i][3]/scale, X[i][2]/scale))
        end
    end
    close(fout)

end


function writeFrameVTK(PL::ParticleList{dim,T}, pressure, frameNum, folder; 
    includeBoundary=false) where {dim, T}

    mkpath(folder)

    if includeBoundary
        I = findall(PL.active)
    else
        I = findall(isequal(:fluid), PL.label)
    end

    xDoc = XMLDocument()

    xRoot = create_root(xDoc, "VTKFile")
    set_attributes(xRoot; 
        type = "UnstructuredGrid", 
        version = "0.1", 
        byte_order = "LittleEndian"
    )

    xGrid = new_child(xRoot, "UnstructuredGrid")

    xPiece = new_child(xGrid, "Piece")
    set_attributes(xPiece; 
        NumberOfPoints = string(length(I)),
        NumberOfCells = "0"
    )

    xPointData = new_child(xPiece, "PointData")
    set_attributes(xPointData;
        Scalars = "density",
        Vectors = "velocity"
    )

    xMass = new_child(xPointData, "DataArray")
    set_attributes(xMass;
        Name = "mass",
        NumberOfComponents = "1",
        format = "ascii",
        type = string(T)
    )

    add_text(xMass, join(@view(PL.mass[I]), " "))
    
    xDensity = new_child(xPointData, "DataArray")
    set_attributes(xDensity;
        Name = "density",
        NumberOfComponents = "1",
        format = "ascii",
        type = string(T)
    )
    add_text(xDensity, join(@view(PL.density[I]), " "))


    xPressure = new_child(xPointData, "DataArray")
    set_attributes(xPressure;
        Name = "pressure",
        NumberOfComponents = "1",
        format = "ascii",
        type = string(T)
    )
    add_text(xPressure, join(@view(pressure[I]), " "))

    
    xRadius = new_child(xPointData, "DataArray")
    set_attributes(xRadius;
        Name = "radius",
        NumberOfComponents = "1",
        format = "ascii",
        type = string(T)
    )
    add_text(xRadius, join(@view(PL.radius[I]), " "))


    xLabel = new_child(xPointData, "DataArray")
    set_attributes(xLabel;
        Name = "label",
        NumberOfComponents = "1",
        format = "ascii",
        type = "UInt8"
    )
    labelIDs = zeros(UInt8, length(I))
    k = 0
    for lbl in sort(unique(PL.label))
        labelIDs[PL.label[I] .=== lbl] .= k
        k += 1
    end
    add_text(xLabel, join(labelIDs, " "))
    

    xVelocity = new_child(xPointData, "DataArray")
    set_attributes(xVelocity;
        Name = "velocity",
        NumberOfComponents = "3",
        format = "ascii",
        type = string(T)
    )
    velocityMat = reinterpret(T, @view(PL.velocity[I]))
    add_text(xVelocity, join(velocityMat, " "))

    xPoints = new_child(xPiece, "Points")
    xPointsChild = new_child(xPoints, "DataArray")
    set_attributes(xPointsChild;
        NumberOfComponents = "3",
        format = "ascii",
        type = string(T)
    )
    pointsMat = reinterpret(T, @view(PL.position[I]))
    add_text(xPointsChild, join(pointsMat, " "))

    xCells = new_child(xPiece, "Cells")
    new_child(xCells, "DataArray")

    filename = "frame.$frameNum.vtu"
    save_file(xDoc, folder * "/" * filename)
    free(xDoc)

    return filename

end

mutable struct VTKWriter
    path
    currentFrame
    masterFile
    collection
    function VTKWriter(path)

        currentFrame = 0

        # open the master file
        doc = XMLDocument()

        root = create_root(doc, "VTKFile")
        set_attributes(root; 
            type = "Collection", 
            version = "0.1",
            byte_order = "LittleEndian"
        )

        collection = new_child(root, "Collection")

        return new(path, currentFrame, doc, collection)

    end
end

function close(V::VTKWriter)
    save_file(V.masterFile, V.path * "/collection.pvd")
    free(V.masterFile)
    return nothing
end

function writeSnapshot(V::VTKWriter, PL::ParticleList, pressure, time; includeBoundary = false)

    # try
    #     filename = writeFrameVTK(PL::ParticleList, pressure, V.currentFrame, V.path; includeBoundary)
    # catch ex
    #     println(ex)
    #     return nothing
    # end

    frameNum = V.currentFrame
    filename = joinpath(V.path, "frame.$frameNum.vtu")

    If = includeBoundary ? Colon() : findall(==(:fluid), PL.label)

    labelIDs = zeros(UInt8, length(If))
    k = 0
    for lbl in sort(unique(PL.label))
        labelIDs[PL.label[If] .=== lbl] .= k
        k += 1
    end

    labelStrings = String.(PL.label[If])
    
    vtk_grid(filename, view(PL.position, If), MeshCell[]) do vtk
        vtk["density"] = @view PL.density[If]
        vtk["mass"] = @view PL.mass[If]
        vtk["pressure"] = @view pressure[If]
        vtk["radius"] = @view PL.radius[If]
        vtk["velocity"] = @view PL.velocity[If]
        vtk["label"] = labelIDs
        vtk["labelString"] = labelStrings
        vtk["time", VTKFieldData()] = Float64(time)
        vtk["dx", VTKFieldData()] = Float64(PL.Δx)
    end

    collectionEntry = new_child(V.collection, "DataSet")
    set_attributes(collectionEntry;
        timestep = string(Float64(time)),
        group = "",
        part = "0",
        file = filename
    )
    V.currentFrame += 1

    return nothing

end

import Dates: format, now
function writeSettingsLog(filename, NL, physics, frame_dt)
    
    PL = NL.particles
    
    open(filename, "w") do file

        # datetime
        print(file, "datetime: ")
        println(file, format(now(), "yyyy/mm/dd HH:MM:SS"))
        
        # number of particles (for each label)
        println(file, "num particles: $(length(PL))")
        for l in unique(PL.label)
            c = count(isequal(l), PL.label)
            println(file, "\t$l: $c")
        end

        # total volume of fluid particles
        V = sum(i -> (PL.label[i] === :fluid ? PL.mass[i]/PL.density[i] : 0), 1:length(PL))
        println(file, "fluid volume: $V")

        # dx
        println(file, "Δx: $(NL.particles.Δx)")

        # h
        println(file, "h: $(NL.h)")

        # physics
        println(file, "g: $(physics.g)")
        println(file, "ρ₀: $(physics.ρ₀)")
        println(file, "viscous? $(physics.viscous)")

        if physics.viscous
            println(file, "μ: $(physics.μ)")
        else
            println(file, "α: $(physics.α)")
        end

        println(file, "c₀: $(physics.c₀)")
        println(file, "δ: $(physics.δ)")

        println(file, "s:")
        for p in physics.s
            println(file, "\t$p")
        end

        println(file, "F: $(physics.F)")

        println(file, "frame_dt: $frame_dt")

    end

end


function writeBoundaryVTK(PL::ParticleList{dim, T}, folder; 
    filename="boundary", boundaryLabels=(:fixedGhost,)) where {dim, T}

    if !isdir(folder)
        mkpath(folder)
    end

    Ib = findall(in(boundaryLabels), PL.label)

    labelIDs = zeros(UInt8, length(Ib))
    k = 0
    for lbl in sort(unique(PL.label))
        labelIDs[PL.label[Ib] .=== lbl] .= k
        k += 1
    end

    labelStrings = String.(PL.label[Ib])

    filepath = joinpath(folder, filename)

    vtk_grid(filepath, view(PL.position, Ib), MeshCell[]) do vtk
        vtk["density"] = @view PL.density[Ib]
        vtk["mass"] = @view PL.mass[Ib]
        vtk["radius"] = @view PL.radius[Ib]
        vtk["label"] = labelIDs
        vtk["labelString"] = labelStrings
    end

    return filename

end

function writeBoundaryVTK_oriented(PL::ParticleList{dim,T}, ∇f, folder) where {dim,T}

    mkpath(folder)

    Ib = findall(isequal(:fixedGhost), PL.label)

    xDoc = XMLDocument()

    xRoot = create_root(xDoc, "VTKFile")
    set_attributes(xRoot; 
        type = "UnstructuredGrid", 
        version = "0.1", 
        byte_order = "LittleEndian"
    )

    xGrid = new_child(xRoot, "UnstructuredGrid")

    xPiece = new_child(xGrid, "Piece")
    set_attributes(xPiece; 
        NumberOfPoints = string(length(Ib)),
        NumberOfCells = "0"
    )

    xPointData = new_child(xPiece, "PointData")
    set_attributes(xPointData;
        Scalars = "density",
        Vectors = "velocity"
    )

    xRadius = new_child(xPointData, "DataArray")
    set_attributes(xRadius;
        Name = "radius",
        NumberOfComponents = "1",
        format = "ascii",
        type = string(T)
    )
    add_text(xRadius, join(@view(PL.radius[Ib]), " "))

    xOrientation = new_child(xPointData, "DataArray")
    set_attributes(xOrientation;
        Name = "orientation",
        NumberOfComponents = "3",
        format = "ascii",
        type = string(T)
    )
    orientation = ∇f.(PL.position)
    orientationMat = [ oi[j] for j in 1:3, oi in orientation ]
    add_text(xOrientation, join(orientationMat, " "))

    xPoints = new_child(xPiece, "Points")
    xPointsChild = new_child(xPoints, "DataArray")
    set_attributes(xPointsChild;
        NumberOfComponents = "3",
        format = "ascii",
        type = string(T)
    )
    pointsMat = reinterpret(T, @view(PL.position[Ib]))
    add_text(xPointsChild, join(pointsMat, " "))

    xCells = new_child(xPiece, "Cells")
    new_child(xCells, "DataArray")

    filename = "boundary.vtu"
    save_file(xDoc, folder * "/" * filename)
    free(xDoc)

    return filename

end




mutable struct JLD2FrameWriter
    saveFile
    currentFrame
    function JLD2FrameWriter(path)
        currentFrame = 0
        saveFile = joinpath(path * "/savedata.jld2")
        jldfile = jldopen(saveFile, "w"; compress=true)
        close(jldfile)
        return new(saveFile, currentFrame)
    end
end

function writeSnapshot(J::JLD2FrameWriter, PL::ParticleList, time; f = pl -> missing, 
    onlySaveSummaryData = false)

    i = J.currentFrame

    try 

        savefile = jldopen(J.saveFile, "a"; compress=true)
        finalizer(close, savefile)

        savefile["$i/time"] = time # this fails if the code is re-run, i.e., if the time entry already exists
        if !onlySaveSummaryData
            savefile["$i/PL"] = PL
        end
        savefile["$i/data"] = f(PL)
        
        close(savefile)

    catch ex
        println(ex)
        return nothing
    end

    J.currentFrame += 1

    return nothing

end

function close(J::JLD2FrameWriter)

    savefile = jldopen(J.saveFile, "a"; compress=true)
    finalizer(close, savefile)

    savefile["numFrames"] = J.currentFrame

    close(savefile)

    return nothing
    
end

#=
function JLD2VTK(path; step=1)

    V = VTKWriter(path)
    
    jldopen(path * "/savedata.jld2", "r") do f

        n = f["numFrames"]

        # if step != 1, the frame nums in the paraview collection will
        # not correspond to frames in the jld2 savefile.
        for i in 0:step:n-1

            time = f["$i/time"]
            PL = f["$i/PL"]
            optargs = f["$i/optargs"]

            writeSnapshot(V, PL, time; optargs...)

        end

    end

    close(V)

    return nothing

end
=#

