using Oceananigans
using Oceananigans.Units
using AquaSkyLES

Nx = Nz = 256
Lz = 3 * 1024
grid = RectilinearGrid(size=(Nx, Nz), x=(0, 2Lz), z=(0, Lz), topology=(Periodic, Flat, Bounded))

ρ₀ = 1   # air density
cₚ = 1e3 # air specific heat
Q = 1000 # heat flux in W / m²
Jθ = Q / (ρ₀ * cₚ)
heating = FluxBoundaryCondition(Jθ)
θ_bcs = FieldBoundaryConditions(bottom=heating)

advection = WENO() #(momentum=WENO(), θ=WENO(), q=WENO(bounds=(0, 1)))
tracers = (:θ, :q)
buoyancy = AquaSkyLES.MoistAirBuoyancy()
model = NonhydrostaticModel(; grid, advection, tracers, buoyancy, boundary_conditions=(; θ=θ_bcs))

Lz = grid.Lz
Δθ = 10 # ᵒC
Tₛ = 20 # ᵒC
θᵢ(x, z) = Tₛ + Δθ * z / Lz + 1e-2 * Δθ * randn()
qᵢ(x, z) = 1e-4 * rand()
set!(model, θ=θᵢ, q=qᵢ)

simulation = Simulation(model, Δt=10, stop_time=1hour)
conjure_time_step_wizard!(simulation, cfl=0.7)
progress(sim) = @info string(iteration(sim), ": ", prettytime(sim))
add_callback!(simulation, progress, IterationInterval(10))

ow = JLD2Writer(model, merge(model.velocities, model.tracers),
                filename = "free_convection.jld2",
                schedule = TimeInterval(1minutes),
                overwrite_existing = true)

simulation.output_writers[:jld2] = ow

run!(simulation)

θt = FieldTimeSeries("free_convection.jld2", "θ")
qt = FieldTimeSeries("free_convection.jld2", "q")
times = qt.times
Nt = length(θt)

using GLMakie, Printf

n = Observable(length(θt))

θn = @lift θt[$n]
qn = @lift qt[$n]
title = @lift "t = $(prettytime(times[$n]))"

fig = Figure(size=(1600, 400), fontsize=22)
axθ = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)")
axq = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)")

fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)

hmθ = heatmap!(axθ, θn, colorrange=(Tₛ, Tₛ+Δθ))
hmq = heatmap!(axq, qn, colorrange=(0, 8e-5), colormap=:magma)

Label(fig[0, 1], "θ", tellwidth=false)
Label(fig[0, 2], "q", tellwidth=false)

Colorbar(fig[2, 1], hmθ, label = "[ᵒC]", vertical=false)
Colorbar(fig[2, 2], hmq, label = "", vertical=false)

fig

record(fig, "free_convection.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end
