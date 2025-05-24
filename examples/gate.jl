# GATE_III
# See: https://github.com/CliMA/AtmosphericProfilesLibrary.jl/blob/main/src/profiles/GATE_III.jl

using Pkg; Pkg.activate(".")
using Oceananigans
using Oceananigans.Units
using Printf
using AquaSkyLES

arch = CPU()

# Siebesma et al (2003) resolution!
# DOI: https://doi.org/10.1175/1520-0469(2003)60<1201:ALESIS>2.0.CO;2
Nx = Ny = 64
Nz = 75
Δx = Δy = 250 # m
Δz = 100 # m

Lx = Δx * Nx
Ly = Δy * Ny
Lz = Δz * Nz

grid = RectilinearGrid(arch,
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       topology = (Periodic, Periodic, Bounded))

using AtmosphericProfilesLibrary                       

FT = eltype(grid)
θ_bomex = AtmosphericProfilesLibrary.Bomex_θ_liq_ice(FT)
q_bomex = AtmosphericProfilesLibrary.Bomex_q_tot(FT)
u_bomex = AtmosphericProfilesLibrary.Bomex_u(FT)

p₀ = 101325 # Pa
θ₀ = θ_bomex(0) # K
reference_state = AquaSkyLES.ReferenceState(base_pressure=p₀, potential_temperature=θ₀)
buoyancy = AquaSkyLES.MoistAirBuoyancy(; reference_state) #, microphysics)

# Simple precipitation scheme from CloudMicrophysics    
using CloudMicrophysics 
using CloudMicrophysics.Microphysics0M: remove_precipitation


θ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(8e-3))
q_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(5.2e-5))

u★ = 0.28 # m/s
@inline u_drag(x, y, t, u, v, u★) = - u★^2 * u / sqrt(u^2 + v^2)
@inline v_drag(x, y, t, u, v, u★) = - u★^2 * v / sqrt(u^2 + v^2)
u_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(u_drag, field_dependencies=(:u, :v), parameters=u★))
v_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(v_drag, field_dependencies=(:u, :v), parameters=u★))

coriolis = FPlane(f=3.76e-5)
uᵍ = Field{Nothing, Nothing, Center}(grid)
vᵍ = Field{Nothing, Nothing, Center}(grid)
uᵍ_bomex = AtmosphericProfilesLibrary.Bomex_geostrophic_u(FT)
vᵍ_bomex = AtmosphericProfilesLibrary.Bomex_geostrophic_v(FT)
set!(uᵍ, z -> uᵍ_bomex(z))
set!(vᵍ, z -> vᵍ_bomex(z))

Fu = Field{Nothing, Nothing, Center}(grid)
Fv = Field{Nothing, Nothing, Center}(grid)

using Oceananigans.Operators: ∂zᶜᶜᶠ, ℑzᵃᵃᶜ
@inline w_dz_ϕ(i, j, k, grid, w, ϕ) = @inbounds w[i, j, k] * ∂zᶜᶜᶠ(i, j, k, grid, ϕ)

@inline function u_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    u_avg = parameters.u_avg
    w_dz_U = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, u_avg)
    return - w_dz_U
end

@inline function v_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    v_avg = parameters.v_avg
    w_dz_V = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, v_avg)
    return - w_dz_V
end

@inline function θ_subsidence(i, j, k, grid, clock, fields, parameters)
    wˢ = parameters.wˢ
    θ_avg = parameters.θ_avg
    w_dz_T = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, θ_avg)
    return - w_dz_T
end

FT = eltype(grid)
microphysics = CloudMicrophysics.Parameters.Parameters0M{FT}(τ_precip=600, S_0=0, qc_0=0.02)
@inline precipitation(x, y, z, t, q, params) = 

@inline function q_subsidence(i, j, k, grid, clock, fields, parameters)
    microphysics = parameters.microphysics
    wˢ = parameters.wˢ
    q_avg = parameters.q_avg
    w_dz_Q = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, wˢ, q_avg)
    q = @inbounds fields.q[i, j, k]
    μ_dqdt = remove_precipitation(microphysics, q, 0)
    return - w_dz_Q + μ_dqdt
end

u_avg = Field{Nothing, Nothing, Center}(grid)
v_avg = Field{Nothing, Nothing, Center}(grid)
θ_avg = Field{Nothing, Nothing, Center}(grid)
q_avg = Field{Nothing, Nothing, Center}(grid)

wˢ = Field{Nothing, Nothing, Face}(grid)
w_bomex = AtmosphericProfilesLibrary.Bomex_subsidence(FT)
set!(wˢ, z -> w_bomex(z))

Fu_subsidence = Forcing(u_subsidence, discrete_form=true, parameters=(; wˢ, u_avg))
Fv_subsidence = Forcing(v_subsidence, discrete_form=true, parameters=(; wˢ, v_avg))
Fθ_subsidence = Forcing(θ_subsidence, discrete_form=true, parameters=(; wˢ, θ_avg))
Fq_subsidence = Forcing(q_subsidence, discrete_form=true, parameters=(; wˢ, q_avg))

set!(Fu, z -> - coriolis.f * vᵍ_bomex(z))
set!(Fv, z -> + coriolis.f * uᵍ_bomex(z))
Fu_geostrophic = Forcing(Fu)
Fv_geostrophic = Forcing(Fv)

u_forcing = (Fu_subsidence, Fu_geostrophic)
v_forcing = (Fv_subsidence, Fv_geostrophic)

drying = Field{Nothing, Nothing, Center}(grid)
dqdt_bomex = AtmosphericProfilesLibrary.Bomex_dqtdt(FT)
set!(drying, z -> dqdt_bomex(z))
Fq_drying = Forcing(drying)
q_forcing = (Fq_1M, Fq_drying, Fq_subsidence)

Fθ_field = Field{Nothing, Nothing, Center}(grid)
dTdt_bomex = AtmosphericProfilesLibrary.Bomex_dTdt(FT)
set!(Fθ_field, z -> dTdt_bomex(1, z))
Fθ_radiation = Forcing(Fθ_field)
θ_forcing = (Fθ_radiation, Fθ_subsidence)

advection = WENO() #(momentum=WENO(), θ=WENO(), q=WENO(bounds=(0, 1)))
tracers = (:θ, :q)
model = NonhydrostaticModel(; grid, advection, buoyancy, coriolis,
                            tracers = (:θ, :q, :qˡ, :qⁱ, :qʳ, :qˢ),
                            forcing = (; q=q_forcing, u=u_forcing, v=v_forcing, θ=θ_forcing),
                            boundary_conditions = (θ=θ_bcs, q=q_bcs, u=u_bcs))

θϵ = 20
qϵ = 1e-2
θᵢ(x, y, z) = θ_bomex(z) + 1e-2 * θϵ * rand()
qᵢ(x, y, z) = q_bomex(z) + 1e-2 * qϵ * rand()
uᵢ(x, y, z) = u_bomex(z)
set!(model, θ=θᵢ, q=qᵢ, u=uᵢ)

simulation = Simulation(model, Δt=10, stop_time=1hours)
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

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, max|u|: (%.2e, %.2e, %.2e)",
                    iteration(sim), prettytime(sim), prettytime(sim.Δt), umax, vmax, wmax)

    msg *= @sprintf(", extrema(q): (%.2e, %.2e), max(qˡ): %.2e, min(δ): %.2e, extrema(θ): (%.2e, %.2e)",
                     qmin, qmax, qˡmax, δmax, θmin, θmax)

    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

# using Oceananigans.Models: ForcingOperation
# Sʳ = ForcingOperation(:q, model)
# outputs = merge(model.velocities, model.tracers, (; T, qˡ, qᵛ★, Sʳ))
outputs = merge(model.velocities, model.tracers, (; T, qˡ, qᵛ★))

ow = JLD2Writer(model, outputs,
                filename = "bomex.jld2",
                schedule = TimeInterval(1minutes),
                overwrite_existing = true)

simulation.output_writers[:jld2] = ow

run!(simulation)

wt = FieldTimeSeries("bomex.jld2", "w")
θt = FieldTimeSeries("bomex.jld2", "θ")
Tt = FieldTimeSeries("bomex.jld2", "T")
qt = FieldTimeSeries("bomex.jld2", "q")
qˡt = FieldTimeSeries("bomex.jld2", "qˡ")
times = qt.times
Nt = length(θt)

using GLMakie, Printf

fig = Figure(size=(1200, 800), fontsize=12)
axθ = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)")
axq = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)")
axT = Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)")
axqˡ = Axis(fig[2, 2], xlabel="x (m)", ylabel="z (m)")
axw = Axis(fig[3, 1], xlabel="x (m)", ylabel="z (m)")

Nt = length(θt)
slider = Slider(fig[4, 1:2], range=1:Nt, startvalue=1)

n = slider.value #Observable(length(θt))
wn = @lift view(wt[$n], :, 1, :)
θn = @lift view(θt[$n], :, 1, :)
qn = @lift view(qt[$n], :, 1, :)
Tn = @lift view(Tt[$n], :, 1, :)
qˡn = @lift view(qˡt[$n], :, 1, :)
title = @lift "t = $(prettytime(times[$n]))"


fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)

Tmin = minimum(Tt)
Tmax = maximum(Tt)
wlim = 0.5 #maximum(abs, wt) / 2
qlim = maximum(abs, qt)
qˡlim = maximum(abs, qˡt) / 2

Tₛ = θ_bomex(0)
Δθ = θ_bomex(Lz) - θ_bomex(0)
hmθ = heatmap!(axθ, θn, colorrange=(Tₛ, Tₛ+Δθ))
hmq = heatmap!(axq, qn, colorrange=(0, qlim), colormap=:magma)
hmT = heatmap!(axT, Tn, colorrange=(Tmin, Tmax))
hmqˡ = heatmap!(axqˡ, qˡn, colorrange=(0, qˡlim), colormap=:magma)
hmw = heatmap!(axw, wn, colorrange=(-wlim, wlim), colormap=:balance)

# Label(fig[0, 1], "θ", tellwidth=false)
# Label(fig[0, 2], "q", tellwidth=false)
# Label(fig[0, 1], "θ", tellwidth=false)
# Label(fig[0, 2], "q", tellwidth=false)

Colorbar(fig[1, 0], hmθ, label = "θ [K]", vertical=true)
Colorbar(fig[1, 3], hmq, label = "q", vertical=true)
Colorbar(fig[2, 0], hmT, label = "T [K]", vertical=true)
Colorbar(fig[2, 3], hmqˡ, label = "qˡ", vertical=true)
Colorbar(fig[3, 0], hmw, label = "w", vertical=true)

fig

record(fig, "bomex.mp4", 1:Nt, framerate=12) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end

using Oceananigans
using Oceananigans.Units
using Printf
using GLMakie
