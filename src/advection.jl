module Advection

using Interpolations: GriddedInterpolation

export advect_rk4

"""
    advect_rk4(dt, pos, wrap, u_prev, v_prev, u_inter, v_inter, u_next, v_next)

RK4 advection kernel

Advect a particle located at position (x, y) with a timestep dt. Velocity fields
are functions returning the velocity evaluated (or interpolated) to a given location.
The velocity fields are specified at the current timestep and at the following timestep.
"""
function advect_rk4(dt, pos, wrap,
                    u_prev::GriddedInterpolation, v_prev::GriddedInterpolation,
                    u_next::GriddedInterpolation, v_next::GriddedInterpolation)
    u_inter(pos) = (u_prev.(pos[:,1], pos[:,2]) + u_next.(pos[:,1], pos[:,2])) / 2
    v_inter(pos) = (v_prev.(pos[:,1], pos[:,2]) + v_next.(pos[:,1], pos[:,2])) / 2

    pos = permutedims(pos, (2, 1))

    vel1 = hcat(u_prev.(pos[:,1], pos[:,2]),
                v_prev.(pos[:,1], pos[:,2]))
    pos1 = wrap(pos .+ 0.5 * vel1 * dt)

    vel2 = hcat(u_inter(pos1), v_inter(pos1))
    pos2 = wrap(pos .+ 0.5 * vel2 * dt)

    vel3 = hcat(u_inter(pos2), v_inter(pos2))
    pos3 = wrap(pos .+ vel3 * dt)

    vel4 = hcat(u_next.(pos3[:,1], pos3[:,2]),
                v_next.(pos3[:,1], pos3[:,2]))

    permutedims(wrap(pos .+ (vel1 .+ 2*vel2 .+ 3*vel3 .+ vel4) / 6*dt), (2, 1))
end

end # module
