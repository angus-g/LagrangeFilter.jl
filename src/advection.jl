module Advection

using Interpolations: GriddedInterpolation

export advect_rk4

function advect_rk4(dt, pos, wrap,
                    u_prev::GriddedInterpolation, v_prev::GriddedInterpolation,
                    u_inter::GriddedInterpolation, v_inter::GriddedInterpolation,
                    u_next::GriddedInterpolation, v_next::GriddedInterpolation)
    vel1 = [u_prev(pos...), v_prev(pos...)]
    pos1 = wrap(pos + 0.5 * vel1 * dt)

    vel2 = [u_inter(pos1...), v_inter(pos1...)]
    pos2 = wrap(pos + 0.5 * vel2 * dt)

    vel3 = [u_inter(pos2...), v_inter(pos2...)]
    pos3 = wrap(pos + vel3 * dt)

    vel4 = [u_next(pos3...), v_next(pos3...)]

    wrap(pos + (vel1 + 2*vel2 + 3*vel3 + vel4) / 6*dt)
end

end # module
