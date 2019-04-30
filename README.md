# LagrangeFilter.jl
Transform data to Lagrangian frame of reference (and back) for filtering, efficiently!

## Algorithm
1. Seed domain with particles, in time and space
2. Advect particles forward and backward in time to cover the full integration period
3. Collect properties of interest along particle paths
4. Convert each particle path to frequency domain and filter
5. Invert Fourier transform back to temporal domain
6. Interpolate back onto the original grid
