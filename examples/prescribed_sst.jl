using Oceananigans
using Oceananigans.Units
using Printf
using AquaSkyLES

Nx = 128
Nz = 128
Ny = 1
Lz = 2 * 1024
grid = RectilinearGrid(size = (Nx, Ny, Nz),
                       x = (0, 2Lz),
                       y = (0, 2Lz),
                       z = (0, Lz),
                       topology = (Periodic, Periodic, Bounded))

p₀ = 101325 # Pa
θ₀ = 285 # K
reference_state = AquaSkyLES.ReferenceState(base_pressure=p₀, potential_temperature=θ₀)
buoyancy = AquaSkyLES.MoistAirBuoyancy(; reference_state)
thermodynamics = buoyancy.thermodynamics
condensation = thermodynamics.condensation

# ρ₀ = AquaSkyLES.base_density(buoyancy) # air density at z=0
# cₚ = buoyancy.thermodynamics.dry_air.heat_capacity

Δz = Lz / Nz # grid spacing

parameters = (; 
    drag_coefficient = 1e-3,
    heat_transfer_coefficient = 1e-3,
    vapor_transfer_coefficient = 1e-3,
    sea_surface_temperature = θ₀ + 10,
    gust_speed = 1e-2, # directly added to friction velocity (i.e. not multiplied by drag coefficient Cᴰ)
    base_air_density = AquaSkyLES.base_density(buoyancy), # air density at z=0,
    thermodynamics,
    condensation
)

# Utility for computing the saturation specific humidity at the sea surface
@inline surface_saturation_specific_humidity(T, parameters) =
    AquaSkyLES.saturation_specific_humidity(T, parameters.base_air_density,
                                            parameters.thermodynamics,
                                            parameters.condensation)


@inline function friction_velocity(i, j, grid, clock, model_fields, parameters)
    Cᴰ = parameters.drag_coefficient
    u = model_fields.u[i, j, 1]
    v = model_fields.v[i, j, 1]
    Δu = u # stationary ocean
    Δv = v # stationary ocean
    return sqrt(Cᴰ * (Δu^2 + Δv^2)) + parameters.gust_speed
end

# Take care to handle U = 0
@inline function x_momentum_flux(i, j, grid, clock, model_fields, parameters)
    u = model_fields.u[i, j, 1]
    v = model_fields.v[i, j, 1]
    u★ = friction_velocity(i, j, grid, clock, model_fields, parameters)
    U = sqrt(u^2 + v^2)
    return - u★^2 * u / U * (U > 0)
end

@inline function y_momentum_flux(i, j, grid, clock, model_fields, parameters)
    u = model_fields.u[i, j, 1]
    v = model_fields.v[i, j, 1]
    u★ = friction_velocity(i, j, grid, clock, model_fields, parameters)
    U = sqrt(u^2 + v^2)
    return - u★^2 * v / U * (U > 0)
end

@inline function temperature_flux(i, j, grid, clock, model_fields, parameters)
    u★ = friction_velocity(i, j, grid, clock, model_fields, parameters)
    θˢ = parameters.sea_surface_temperature
    Cᴰ = parameters.drag_coefficient
    Cᴴ = parameters.heat_transfer_coefficient
    Δθ = model_fields.θ[i, j, 1] - θˢ
    # Using the scaling argument: u★ θ★ = Cᴴ * U * Δθ
    θ★ = Cᴴ / sqrt(Cᴰ) * Δθ
    return - u★ * θ★
end

@inline function vapor_flux(i, j, grid, clock, model_fields, parameters)
    u★ = friction_velocity(i, j, grid, clock, model_fields, parameters)
    θˢ = parameters.sea_surface_temperature
    qˢ = surface_saturation_specific_humidity(θˢ, parameters)
    Cᴰ = parameters.drag_coefficient
    Cᵛ = parameters.vapor_transfer_coefficient
    Δq = model_fields.q[i, j, 1] - qˢ
    # Using the scaling argument: u★ q★ = Cᵛ * U * Δq
    q★ = Cᵛ / sqrt(Cᴰ) * Δq 
    return - u★ * q★
end

u_surface_flux = FluxBoundaryCondition(x_momentum_flux; discrete_form=true, parameters)
v_surface_flux = FluxBoundaryCondition(y_momentum_flux; discrete_form=true, parameters)
θ_surface_flux = FluxBoundaryCondition(temperature_flux; discrete_form=true, parameters)
q_surface_flux = FluxBoundaryCondition(vapor_flux; discrete_form=true, parameters)

u_bcs = FieldBoundaryConditions(bottom=u_surface_flux)
v_bcs = FieldBoundaryConditions(bottom=v_surface_flux)
θ_bcs = FieldBoundaryConditions(bottom=θ_surface_flux)
q_bcs = FieldBoundaryConditions(bottom=q_surface_flux)

advection = WENO() #(momentum=WENO(), θ=WENO(), q=WENO(bounds=(0, 1)))
tracers = (:θ, :q)
model = NonhydrostaticModel(; grid, advection, buoyancy,
                            tracers = (:θ, :q),
                            boundary_conditions = (u=u_bcs, v=v_bcs, θ=θ_bcs, q=q_bcs))

Lz = grid.Lz
Δθ = 2.5 # K
Tₛ = reference_state.θ # K
θᵢ(x, y, z) = Tₛ + Δθ * z / Lz + 1e-2 * Δθ * randn()
qᵢ(x, y, z) = 1e-2 + 1e-5 * rand()
set!(model, θ=θᵢ, q=qᵢ)

simulation = Simulation(model, Δt=10, stop_time=4hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

T = AquaSkyLES.TemperatureField(model)
qˡ = AquaSkyLES.CondensateField(model, T)
qᵛ★ = AquaSkyLES.SaturationField(model, T)
δ = Field(model.tracers.q - qᵛ★)


function progress(sim)
    compute!(T)
    compute!(qˡ)
    compute!(δ)
    q = sim.model.tracers.q
    θ = sim.model.tracers.θ
    u, v, w = sim.model.velocities

    umax = maximum(u)
    vmax = maximum(v)
    wmax = maximum(w)

    qmin = minimum(q)
    qmax = maximum(q)
    qˡmax = maximum(qˡ)
    δmax = maximum(δ)

    θmin = minimum(θ)
    θmax = maximum(θ)

    msg = @sprintf("Iter: %d, t = %s, max|u|: (%.2e, %.2e, %.2e)",
                    iteration(sim), prettytime(sim), umax, vmax, wmax)

    msg *= @sprintf(", extrema(q): (%.2e, %.2e), max(qˡ): %.2e, min(δ): %.2e, extrema(θ): (%.2e, %.2e)",
                     qmin, qmax, qˡmax, δmax, θmin, θmax)

    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

outputs = merge(model.velocities, model.tracers, (; T, qˡ, qᵛ★))

ow = JLD2Writer(model, outputs,
                filename = "prescribed_sst_convection.jld2",
                schedule = TimeInterval(2minutes),
                overwrite_existing = true)

simulation.output_writers[:jld2] = ow

run!(simulation)

θt = FieldTimeSeries("prescribed_sst_convection.jld2", "θ")
Tt = FieldTimeSeries("prescribed_sst_convection.jld2", "T")
qt = FieldTimeSeries("prescribed_sst_convection.jld2", "q")
qˡt = FieldTimeSeries("prescribed_sst_convection.jld2", "qˡ")
times = qt.times
Nt = length(θt)

using GLMakie, Printf

n = Observable(length(θt))

θn = @lift θt[$n]
qn = @lift qt[$n]
Tn = @lift Tt[$n]
qˡn = @lift qˡt[$n]
title = @lift "t = $(prettytime(times[$n]))"

fig = Figure(size=(800, 400), fontsize=12)
axθ = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)")
axq = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)")
axT = Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)")
axqˡ = Axis(fig[2, 2], xlabel="x (m)", ylabel="z (m)")

fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)

Tmin = minimum(Tt)
Tmax = maximum(Tt)

hmθ = heatmap!(axθ, θn, colorrange=(Tₛ, Tₛ+Δθ))
hmq = heatmap!(axq, qn, colorrange=(0.97e-2, 1.05e-2), colormap=:magma)
hmT = heatmap!(axT, Tn, colorrange=(Tmin, Tmax))
hmqˡ = heatmap!(axqˡ, qˡn, colorrange=(0, 1.5e-3), colormap=:magma)

# Label(fig[0, 1], "θ", tellwidth=false)
# Label(fig[0, 2], "q", tellwidth=false)
# Label(fig[0, 1], "θ", tellwidth=false)
# Label(fig[0, 2], "q", tellwidth=false)

Colorbar(fig[1, 0], hmθ, label = "θ [K]", vertical=true)
Colorbar(fig[1, 3], hmq, label = "q", vertical=true)
Colorbar(fig[2, 0], hmT, label = "T [K]", vertical=true)
Colorbar(fig[2, 3], hmqˡ, label = "qˡ", vertical=true)

fig

record(fig, "prescribed_sst.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end
