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
    u_inter(pos) = (u_prev(pos...) + u_next(pos...)) / 2
    v_inter(pos) = (v_prev(pos...) + v_next(pos...)) / 2

    vel1 = [u_prev(pos...), v_prev(pos...)]
    pos1 = wrap(pos + 0.5 * vel1 * dt)

    vel2 = [u_inter(pos1), v_inter(pos1)]
    pos2 = wrap(pos + 0.5 * vel2 * dt)

    vel3 = [u_inter(pos2), v_inter(pos2)]
    pos3 = wrap(pos + vel3 * dt)

    vel4 = [u_next(pos3...), v_next(pos3...)]

    wrap(pos + (vel1 + 2*vel2 + 3*vel3 + vel4) / 6*dt)
end

end # module
