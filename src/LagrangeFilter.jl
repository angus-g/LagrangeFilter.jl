using FFTW, Interpolations, LinearAlgebra, NCDatasets, NearestNeighbors
import Distributions.Uniform

using Distributed, SharedArrays
using ProgressMeter

function filter(fname_u, fname_v; np::Int=nothing, nt::Int=nothing, fname_part="particles.nc")
    ds_u = Dataset(fname_u, "r")
    ds_v = Dataset(fname_v, "r")

    ## check dimensions are compatible
    if ds_u["X"] != ds_v["X"] || ds_u["Y"] != ds_v["Y"] || ds_u["T"] != ds_v["T"]
        println("Error: velocity dataset dimensions don't match")
        exit(1)
    end

    # load coordinates and velocity variables without performing CF conversion
    x = nomissing(ds_u["X"][:]); y = nomissing(ds_u["Y"][:])
    var_u = variable(ds_u, "U"); var_v = variable(ds_v, "V")

    # get velocity timestep
    dt = ds_u["T"][2] - ds_u["T"][1]

    if nt == nothing
        nt = length(ds_u["T"])
    end

    ## generate random initial seeds
    if np == nothing
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

    # load first velocity fields
    u_next = var_u[:,:,1,1]
    v_next = var_v[:,:,1,1]

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
            u_next = var_u[:,:,t,1]; v_next = var_v[:,:,t,1]
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

function interp_paths(particles, fname_data, var)
    ds = Dataset(fname_data, "r")
    data = variable(ds, var)
    x = ds["X"][:]; y = ds["Y"][:]

    _, np, nt = size(particles)
    #out_interp = Array{Float64}(undef, np, nt)
    out_interp = SharedArray{Float64}(np, nt)

    time_load = 0.
    time_interp = 0.

    @showprogress 1 "Path interpolation..." for t = 1:nt
        time_load += @elapsed begin
            itp = interpolate((x, y), data[:,:,t,1], Gridded(Linear()))
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
