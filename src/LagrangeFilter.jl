using GridInterpolations, NCDatasets
import Distributions.Uniform

## extend grid interpolation to work on array views
import GridInterpolations.interpolate
using LinearAlgebra

function interpolate(grid::AbstractGrid, data, x::AbstractVector)
    index, weight = interpolants(grid, x)
    dot(data[index], weight)
end
# also fix matrices (re-mapping everything to Float64 is very very slow...)
function interpolate(grid::AbstractGrid, data::Matrix, x::AbstractVector)
    index, weight = interpolants(grid, x)
    dot(data[index], weight)
end

function filter(fname_u, fname_v)
    ds_u = Dataset(fname_u, "r")
    ds_v = Dataset(fname_v, "r")

    ## check dimensions are compatible
    if ds_u["X"] != ds_v["X"] || ds_u["Y"] != ds_v["Y"] || ds_u["T"] != ds_v["T"]
        println("Error: velocity dataset dimensions don't match")
        exit(1)
    end

    # get velocity timestep
    dt = ds_u["T"][2] - ds_u["T"][1]

    ## generate random initial seeds
    num_seeds = Int(length(ds_u["X"]) * length(ds_u["Y"]) / 16)

    num_seeds = 1000
    x_dist = Uniform(ds_u["X"][1], ds_u["X"][end])
    y_dist = Uniform(ds_u["Y"][1], ds_u["Y"][end])

    seeds = hcat(rand(x_dist, num_seeds),
                 rand(y_dist, num_seeds))

    ## save gif of seed positions
    # anim = @animate for i = 1:16
    #     scatter(seed_x[i:16:end], seed_y[i:16:end], markersize=2, markerstrokewidth=0,
    #             xlims = (ds_u["X"][1], ds_u["X"][end]),
    #             ylims = (ds_u["Y"][1], ds_u["Y"][end]))
    # end
    # gif(anim, "seeds.gif", fps = 5)

    ## set up output file
    ds_out = Dataset("particles.nc", "c")
    ds_out.dim["d"] = 2
    ds_out.dim["n"] = num_seeds
    ds_out.dim["time"] = Inf

    part_arr = defVar(ds_out, "particles", Float64, ("time", "n", "d"))

    grid = RectangleGrid(ds_u["X"], ds_u["Y"])
    left = ds_u["X"][1]
    width = ds_u["X"][end] - left

    # load first velocity fields
    u_next = ds_u["U"][:,:,1,1]
    v_next = ds_v["V"][:,:,1,1]

    ## advect particles (rk4)
    anim = @animate for t = 1:length(ds_u["T"])
        println(t)

        scatter(seeds[:,1], seeds[:,2],
                markersize=2, markerstrokewidth=0,
                xlims = (ds_u["X"][1], ds_u["X"][end]),
                ylims = (ds_u["Y"][1], ds_u["Y"][end]))

        # update velocity
        u = u_next; v = v_next
        u_next = ds_u["U"][:,:,t,1]; v_next = ds_v["V"][:,:,t,1]
        u_inter = (u + u_next) / 2; v_inter = (v + v_next) / 2

        vel_prev = (u, v)
        vel_inter = (u_inter, v_inter)
        vel_next = (u_next, v_next)

        velocity_at(vels, pos) = [interpolate(grid, vels[1], pos),
                                  interpolate(grid, vels[2], pos)]
        period_x(pos) = [((pos[1] - left) + width) % width + left, pos[2]]

        # update particle positions
        for p in 1:num_seeds
            pos = seeds[p,:]
            
            # evaluate at (t,x,y)
            vel1 = velocity_at(vel_prev, pos)
            pos1 = period_x(pos + 0.5 * vel1 * dt)

            # evaluate at (t+.5,x',y')
            vel2 = velocity_at(vel_inter, pos1)
            pos2 = period_x(pos + 0.5 * vel2 * dt)

            vel3 = velocity_at(vel_inter, pos2)
            pos3 = period_x(pos + vel3 * dt)

            vel4 = velocity_at(vel_next, pos3)

            seeds[p,:] += (vel1 + 2*vel2 + 2*vel3 + vel4) / 6 * dt
            seeds[p,:] = period_x(seeds[p,:])
        end

        part_arr[t,:,:] = seeds
    end
    gif(anim, "particles.gif", fps=15)

    close(ds_out)

    close(ds_u)
    close(ds_v)
end
    
# using Plots
# pyplot(leg=false)

# if length(ARGS) < 2
#     println("Usage: julia lfilt.jl <U_file> <V_file>")
#     exit(1)
# end
