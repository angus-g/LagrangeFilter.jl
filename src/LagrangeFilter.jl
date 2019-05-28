module LagrangeFilter

using FFTW, Interpolations, LinearAlgebra, NCDatasets, NearestNeighbors
import Distributions.Uniform

using Distributed, SharedArrays
using ProgressMeter

"""
    read_var_shmem(fname, var, t)

Read a variable from a netCDF file using shared-memory parallelism.

NetCDF offers parallel access of netCDF4 files (through the HDF5)
library, but only through MPI. However, we don't want to use this
paradigm for other parts of the code due to the communication
overhead. However, the cost of decompressing datasets is quite high as
it is single-threaded. Ideally, we could read separate chunks using
different threads to perform this decompression in parallel. But,
because HDF5 (and by extension, netCDF) locks opened files for
single-threaded access, we can't do this.

Instead,  this function  uses a  SharedArray (giving  us normal  array
semantics over  a shared  memory-backed storage)  and populates  it in
parallel,  where each  thread opens  its own  handle to  the specified
netCDF file to get around the locking restriction.
"""
function read_var_shmem(fname, var, t)
    # open the dataset on the main thread to read metadata
    d = Dataset(fname, "r")
    meta = d[var]

    # create shared array
    s = SharedArray{Float64}(size(meta, 1), size(meta, 2))

    # read chunking info and decompose across threads
    # note we only chunk in the second dimension, since we leverage
    # the contiguity of the first dimension for better cache access
    ch = chunking(meta)
    chunk_size = 1 # assume not chunked
    if (ch[1] == :chunked)
        chunk_size = ch[2][2]
    end
    n_chunks = size(meta, 2) / chunk_size

    splits = [round(Int, x) for x in range(0, stop=n_chunks, length=length(procs(s))+1)]
    ranges = [splits[i]*chunk_size + 1:splits[i+1]*chunk_size for i in 1:length(procs(s))]

    @sync @distributed for r in ranges
        Dataset(fname, "r") do d
            s[:,r] = d[var][:,r,t,1]
        end
    end

    close(d)

    s
end

"""
    fdiff(var, dx, dim)

Partial derivative of an array along a dimension.

This computes the fourth-order first derivative of a matrix along the
specified dimension (by index). It assumes non-periodicity, and shifts
the finite differencing stencil at the edges.
"""
function fdiff(var, dx, dim::Integer)
    d = similar(v)

    # this is a kind of ugly way to do the iteration, but it lets us work on 1D
    # slices in the simple case
    for (line, diff) in zip(eachslice(var; dims=3-dim), eachslice(d; dims=3-dim))
        # left edge
        diff[1] = [-25, 48, -36, 16, -3] ⋅ line[1:5] / (12 * dx)
        diff[2] = [-3, -10,  18, -6,  1] ⋅ line[1:5] / (12 * dx)

        # usual centered difference for the interior
        for i = 3:length(diff)-2
            diff[i] = [1, -8, 0, 8, -1] ⋅ line[i-2:i+2] / (12 * dx)
        end

        # right edge (mirrored and negated coefficients)
        diff[end-1] = [-1, 6, -18, 10, 3] ⋅ line[end-4:end] / (12 * dx)
        diff[end] = [3, -16, 36, -48, 25] ⋅ line[end-4:end] / (12 * dx)
    end

    d
end

function filter(fname_u, fname_v; np::Int=-1, nt::Int=-1, fname_part="particles.nc")
    ds_u = Dataset(fname_u, "r")
    ds_v = Dataset(fname_v, "r")

    ## check dimensions are compatible
    if ds_u["X"] != ds_v["X"] || ds_u["Y"] != ds_v["Y"] || ds_u["T"] != ds_v["T"]
        println("Error: velocity dataset dimensions don't match")
        exit(1)
    end

    # load coordinates and velocity variables without performing CF conversion
    x = nomissing(ds_u["X"][:]); y = nomissing(ds_u["Y"][:])

    # get velocity timestep
    dt = ds_u["T"][2] - ds_u["T"][1]

    if nt == -1
        nt = length(ds_u["T"])
    end

    ## generate random initial seeds
    if np == -1
        num_seeds = Int(length(x) * length(x) / 16)
    else
        num_seeds = np
    end

    x_dist = Uniform(x[1], x[end])
    y_dist = Uniform(y[1], y[end])

    ## single array for seed positions, transpose so that x/y are adjacent
    seeds = SharedArray{Float64}(2, num_seeds)
    seeds[1,:] = rand(x_dist, num_seeds)
    seeds[2,:] = rand(y_dist, num_seeds)

    ## set up output file
    ds_out = Dataset(fname_part, "c")
    ds_out.dim["d"] = 2
    ds_out.dim["n"] = num_seeds
    ds_out.dim["time"] = Inf

    # output variable (match layout of seeds)
    part_arr = defVar(ds_out, "particles", Float64, ("d", "n", "time"))

    left = x[1]; width = x[end] - left

    var_u(t) = read_var_shmem(fname_u, "U", t)
    var_v(t) = read_var_shmem(fname_v, "V", t)

    # load first velocity fields
    u_next = var_u(1)
    v_next = var_v(1)

    # convenience function to create an interpolator for a velocity field
    interpolator(u) = interpolate((x, y), u, Gridded(Linear()))

    time_load = 0.
    time_interp = 0.
    time_advect = 0.
    time_write = 0.

    ## advect particles (rk4)
    @showprogress 1 "Particle advection..." for t = 1:nt
        # update velocity
        time_load += @elapsed begin
            u = u_next; v = v_next
            u_next = var_u(t); v_next = var_v(t)
            # in-between timesteps
            u_inter = (u + u_next) / 2; v_inter = (v + v_next) / 2
        end

        # interpolators
        time_interp += @elapsed begin
            vel_prev = (interpolator(u), interpolator(v))
            vel_inter = (interpolator(u_inter), interpolator(v_inter))
            vel_next = (interpolator(u_next), interpolator(v_next))
        end

        velocity_at(vels, pos) = [vels[1](pos...)
                                  vels[2](pos...)]
        # interpret a position as periodic in the x direction
        period_x(pos) = [((pos[1] - left) + width) % width + left, pos[2]]

        # update particle positions
        time_advect += @elapsed begin
            @sync @distributed for p = 1:num_seeds
                pos = seeds[:,p]
                
                # evaluate at (t,x,y)
                vel1 = velocity_at(vel_prev, pos)
                pos1 = period_x(pos + 0.5 * vel1 * dt)

                # evaluate at (t+.5,x',y')
                vel2 = velocity_at(vel_inter, pos1)
                pos2 = period_x(pos + 0.5 * vel2 * dt)

                vel3 = velocity_at(vel_inter, pos2)
                pos3 = period_x(pos + vel3 * dt)

                vel4 = velocity_at(vel_next, pos3)

                seeds[:,p] += (vel1 + 2*vel2 + 2*vel3 + vel4) / 6 * dt
                seeds[:,p] = period_x(seeds[:,p])
            end
        end

        time_write += @elapsed part_arr[:,:,t] = seeds
    end

    close(ds_out)
    close(ds_u)
    close(ds_v)

    time_load, time_interp, time_advect, time_write
end

"""
    interp_paths(particles, fname_data, var)

Interpolate variable data onto particle paths.

For a timeseries of particle positions (2 × np × nt), interpolate the
variable with name var from the specified file.
"""
function interp_paths(particles, fname_data, var)
    ds = Dataset(fname_data, "r")
    x = ds["X"][:]; y = ds["Y"][:]
    close(ds)

    _, np, nt = size(particles)
    out_interp = SharedArray{Float64}(np, nt)

    time_load = 0.
    time_interp = 0.

    @showprogress 1 "Path interpolation..." for t = 1:nt
        time_load += @elapsed begin
            data = read_var_shmem(fname_data, var, t)
            itp = interpolate((x, y), data, Gridded(Linear()))
        end

        part_slice = particles[:,:,t]
        
        time_interp += @elapsed begin
            @sync @distributed for p = 1:np
                out_interp[p,t] = itp(part_slice[:,p]...)
            end
        end
    end

    close(ds)
    out_interp, time_load, time_interp
end

"""
    filter_paths(paths, dt, min_freq, butterworth)

High-pass filter Langrangian data along particle paths.

For an array of data along particle paths (np × nt), with timestep
dt, highpass filter each individual path with a given cutoff frequency
and Butterworth filter coefficient.
"""
function filter_paths(paths, dt, min_freq, butterworth)
    nt = size(paths, 2)

    # construct list frequencies for FFT components
    ω = 2π/dt * range(0, 1, length=nt)

    # butterworth highpass filter
    mask = similar(ω)
    @. mask = 1 - 1/sqrt(1 + (ω/min_freq)^(2*butterworth))
    # trim to real coefficients only
    mask = mask[1:floor(Int, nt/2+1)]

    # perform actual filtering -- along time dimension
    # need to broadcast out mask
    irfft(rfft(paths, 2) .* reshape(mask, 1, :), nt, 2)
end

"""
    reinterp_grid(positions, data, points, nx, ny)

Re-interpolate one timestep of Lagrangian data onto an Eulerian grid.

For particles located at the given positions through time (2 × np),
and data sampled at those particles, re-sample onto the grid specified
by points.
"""
function reinterp_grid(positions, data, points, nx, ny)
    kdt = KDTree(positions)
    out = SharedArray{Float64}(nx, ny)

    @sync @distributed for i in eachindex(out)
        idxs, dists = knn(kdt, points[i,:], 6)
        weights = 1 / dists
        weights /= sum(weights)

        out[i] = dot(data[idxs], weights)
    end

    out
end

end # module
