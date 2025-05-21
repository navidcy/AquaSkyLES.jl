using ..AquaSkyLES:
    AtmosphereThermodynamics,
    ReferenceState,
    reference_pressure,
    reference_density,
    mixture_gas_constant,
    mixture_heat_capacity,
    dry_air_gas_constant

using Oceananigans
using Oceananigans.Advection: Centered, adapt_advection_order
using Oceananigans.Grids: ZDirection
using Oceananigans.Models: AbstractModel
using Oceananigans.Architectures: AbstractArchitecture
using Oceananigans.TimeSteppers: TimeStepper
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions
using Oceananigans.Solvers: FourierTridiagonalPoissonSolver
using Oceananigans.Utils: launch!

using KernelAbstractions: @kernel, @index

struct AnelasticReferenceState{FT, F}
    constants :: ReferenceState{FT}
    pressure :: F
    density :: F
end

struct AnelasticParcelState{FT}
    θ :: FT
    q :: FT
    pʳ :: FT
    ρʳ :: FT
    Π :: FT
end

@inline function AnelasticParcelState(θ, q, pʳ, ρʳ, thermo::AtmosphereThermodynamics)
    Rᵐ = mixture_gas_constant(q, thermo)
    cᵖᵐ = mixture_heat_capacity(q, thermo)
    inv_ϰᵐ = Rᵐ / cᵖᵐ
    Π = @inbounds (pʳ / pʳ[1])^inv_ϰᵐ
    return AnelasticParcelState(θ, q, pʳ, ρʳ, Π)
end

function AnelasticReferenceState(grid, ref_constants, thermo)
    pʳ = Field{Center, Nothing, Nothing}(grid)
    ρʳ = Field{Center, Nothing, Nothing}(grid)
    set!(pʳ, z -> reference_pressure(z, ref_constants, thermo))
    set!(ρʳ, z -> reference_density(z, ref_constants, thermo))
    return AnelasticReferenceState(ref_constants, pʳ, ρʳ)
end

@inline function anelastic_specific_volume(i, j, k, grid, reference_state, temperature, specific_humidity, thermo)
    @inbounds begin
        q  = specific_humidity[i, j, k]
        pʳ = reference_state.pressure[i, j, k]
        T = temperature[i, j, k]
    end

    Rᵐ = mixture_gas_constant(q, thermo)

    return Rᵐ * T / pʳ
end

@inline function reference_specific_volume(i, j, k, grid, reference_state, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    pʳ = @inbounds reference_state.pressure[i, j, k]
    θʳ = reference_state.constants.θ
    return Rᵈ * θʳ / pʳ
end


field_names(::AnelasticReferenceState, tracer_names) = (:ρu, :ρv, :ρw, :ρθ, :ρq, tracer_names...)

function collect_prognostic_fields(::AnelasticReferenceState,
                                   density,
                                   momentum,
                                   potential_temperature_density,
                                   absolute_humidity,
                                   condensates,
                                   tracers)

    thermodynamic_variables = (ρθ=potential_temperature_density, ρq=absolute_humidity)

    return merge(momentum, thermodynamic_variables, condensates, tracers)
end

function default_reference_state(grid, thermo)
    FT = eltype(grid)
    base_pressure = convert(FT, 101325)
    potential_temperature = convert(FT, 288)
    constants = ReferenceState(base_pressure, potential_temperature)
    return AnelasticReferenceState(grid, constants, thermo)
end

function materialize_momentum_and_velocities(reference_state::AnelasticReferenceState, grid, boundary_conditions)
    ρu = XFaceField(grid, boundary_conditions=boundary_conditions.ρu)
    ρv = YFaceField(grid, boundary_conditions=boundary_conditions.ρv)
    ρw = ZFaceField(grid, boundary_conditions=boundary_conditions.ρw)
    momentum = (; ρu, ρv, ρw)

    u = XFaceField(grid)
    v = YFaceField(grid)
    w = ZFaceField(grid)
    velocities = (; u, v, w)

    return velocities, momentum
end

materialize_condenstates(microphysics, grid) = NamedTuple() #(; qˡ=CenterField(grid), qᵛ=CenterField(grid))
materialize_density(reference_state, grid) = CenterField(grid)

struct WarmPhaseSaturationAdjustment end

tupleit(t::Tuple) = t
tupleit(t) = tuple(t)

mutable struct AtmosphereModel{Re, Ar, Ts, Gr, Cl, Th, De, Mo, Pd, Wa, En, Hu,
                               Te, Pr, Pp, So, Ve, Tr, Ad, Co, Mi, Cn} <: AbstractModel{Ts, Ar}
    architecture :: Ar
    grid :: Gr
    clock :: Cl
    reference_state :: Re
    thermodynamics :: Th
    density :: De
    momentum :: Mo
    potential_temperature_density :: Pd
    absolute_humidity :: Wa
    potential_temperature :: En # TODO: generalize energy formulation
    specific_humidity :: Hu
    temperature :: Te
    pressure :: Pr
    hydrostatic_pressure_anomaly :: Pp
    pressure_solver :: So
    velocities :: Ve
    tracers :: Tr
    advection :: Ad
    coriolis :: Co
    microphysics :: Mi
    condensates :: Cn
    timestepper :: Ts
end

function AtmosphereModel(grid;
                         clock = Clock(grid),
                         thermodynamics = AtmosphereThermodynamics(eltype(grid)),
                         reference_state = default_reference_state(grid, thermodynamics),
                         specific_humidity = CenterField(grid),
                         tracers = tuple(),
                         coriolis = nothing,
                         boundary_conditions = NamedTuple(),
                         advection = WENO(order=5),
                         microphysics = WarmPhaseSaturationAdjustment(),
                         timestepper = :RungeKutta3)

    arch = grid.architecture
    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)

    hydrostatic_pressure_anomaly = CenterField(grid)
    pressure = CenterField(grid)

    # Next, we form a list of default boundary conditions:
    names = field_names(reference_state, tracers)
    default_boundary_conditions = NamedTuple{names}(FieldBoundaryConditions() for _ in names)
    boundary_conditions = merge(default_boundary_conditions, boundary_conditions)
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, names)

    density = materialize_density(reference_state, grid)
    velocities, momentum = materialize_momentum_and_velocities(reference_state, grid, boundary_conditions)
    tracers = NamedTuple(n => CenterField(grid, boundary_conditions=boundary_conditions[n]) for name in tracers)
    condensates = materialize_condenstates(microphysics, grid)
    advection = adapt_advection_order(advection, grid)

    potential_temperature_density = CenterField(grid, boundary_conditions=boundary_conditions.ρθ)
    absolute_humidity = CenterField(grid, boundary_conditions=boundary_conditions.ρq)
    potential_temperature = CenterField(grid)
    specific_humidity = CenterField(grid)
    temperature = CenterField(grid)

    prognostic_fields = collect_prognostic_fields(reference_state,
                                                  density,
                                                  momentum,
                                                  potential_temperature_density,
                                                  absolute_humidity,
                                                  condensates,
                                                  tracers)

    timestepper = TimeStepper(timestepper, grid, prognostic_fields)
    pressure_solver = FourierTridiagonalPoissonSolver(grid, tridiagonal_direction=ZDirection())

    model = AtmosphereModel(arch,
                            grid,
                            clock,
                            reference_state,
                            thermodynamics,
                            density,
                            momentum,
                            potential_temperature_density,
                            absolute_humidity,
                            potential_temperature,
                            specific_humidity,
                            temperature,
                            pressure,
                            hydrostatic_pressure_anomaly,
                            pressure_solver,
                            velocities,
                            tracers,
                            advection,
                            coriolis,
                            microphysics,
                            condensates,
                            timestepper)

    update_state!(model)

    return model
end
   

using Oceananigans.Utils: prettytime, ordered_dict_show, prettykeys
using Oceananigans.TurbulenceClosures: closure_summary

function Base.summary(model::AtmosphereModel)
    A = nameof(typeof(model.grid.architecture))
    G = nameof(typeof(model.grid))
    return string("AtmosphereModel{$A, $G}",
                  "(time = ", prettytime(model.clock.time), ", iteration = ", model.clock.iteration, ")")
end

function Base.show(io::IO, model::AtmosphereModel)
    TS = nameof(typeof(model.timestepper))
    tracernames = prettykeys(model.tracers)

    print(io, summary(model), "\n",
        "├── grid: ", summary(model.grid), "\n",
        "├── timestepper: ", TS, "\n",
        "├── advection scheme: ", summary(model.advection), "\n",
        "├── tracers: ", tracernames, "\n",
        "└── coriolis: ", summary(model.coriolis))
end

#####
##### update state
#####

using Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.TimeSteppers: update_state!
import Oceananigans: fields, prognostic_fields

const AnelasticModel = AtmosphereModel{<:AnelasticReferenceState}

function prognostic_fields(model::AnelasticModel)
    thermodynamic_fields = (ρθ=model.potential_temperature_density, ρq=model.absolute_humidity)
    return merge(model.momentum, thermodynamic_fields, model.condensates, model.tracers)
end

fields(model::AnelasticModel) = prognostic_fields(model)

function update_state!(model::AnelasticModel)
    fill_halo_regions!(prognostic_fields(model), model.clock, fields(model), async=true)
    compute_auxiliary_variables!(model)
    update_hydrostatic_pressure!(model)
    compute_tendencies!(model)
    return nothing
end

compute_tendencies!(model::AnelasticModel) = nothing

function compute_auxiliary_variables!(model)
    grid = model.grid
    arch = grid.architecture
    velocities = model.velocities
    ref_state = model.reference_state
    momentum = model.momentum

    launch!(arch, grid, :xyz, _compute_velocities!, velocities, ref_state, momentum)

    launch!(arch, grid, :xyz,
            _compute_auxiliary_thermodynamic_variables!,
            model.temperature,
            model.potential_temperature,
            model.specific_humidity,
            model.thermodynamics,
            ref_state,
            model.potential_temperature_density,
            model.absolute_humidity)

    return nothing
end

@kernel function _compute_velocities!(velocities, ref_state, momentum)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρʳ = ref_state.density[i, j, k]
        ρu = momentum.ρu[i, j, k]
        ρv = momentum.ρv[i, j, k]
        ρw = momentum.ρw[i, j, k]

        velocities.u[i, j, k] = ρu / ρʳ
        velocities.v[i, j, k] = ρv / ρʳ
        velocities.w[i, j, k] = ρw / ρʳ
    end
end

@kernel function _compute_auxiliary_thermodynamic_variables!(temperature,
                                                             potential_temperature,
                                                             specific_humidity,
                                                             thermo,
                                                             ref_state,
                                                             potential_temperature_density,
                                                             absolute_humidity)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρʳ = ref_state.density[i, j, k]
        pʳ = ref_state.pressure[i, j, k]
        ρθ = potential_temperature_density[i, j, k]
        ρq = absolute_humidity[i, j, k]

        potential_temperature[i, j, k] = θ = ρθ / ρʳ
        specific_humidity[i, j, k] = q = ρq / ρʳ
    end

    # Saturation adjustment
    Ψ = AnelasticParcelState(θ, q, ρʳ, pʳ, thermo)
    T = anelastic_temperature(Ψ, thermo)
    @inbounds temperature[i, j, k] = T
end
#####
##### update pressure
#####

using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ, ℑzᵃᵃᶠ
using Oceananigans.ImmersedBoundaries: PartialCellBottom, ImmersedBoundaryGrid
using Oceananigans.Grids: topology
using Oceananigans.Grids: XFlatGrid, YFlatGrid

const c = Center()
const f = Face()

@inline function anelastic_buoyancy(i, j, k, grid, ref_state, temperature, specific_humidity, thermo)
    α = anelastic_specific_volume(i, j, k, grid, ref_state, temperature, specific_humidity, thermo)
    αʳ = reference_specific_volume(i, j, k, grid, ref_state, thermo)
    g = thermo.gravitational_acceleration
    return g * (α - αʳ) / αʳ
end

"""
Update the hydrostatic pressure perturbation pHY′. This is done by integrating
the `buoyancy_perturbationᶜᶜᶜ` downwards:

    `pHY′ = ∫ buoyancy_perturbationᶜᶜᶜ dz` from `z=0` down to `z=-Lz`
"""
@kernel function _update_hydrostatic_pressure!(hydrostatic_pressure_anomaly, grid, ref_state, temperature, specific_humidity, thermo)
    i, j = @index(Global, NTuple)
    pₕ′ = hydrostatic_pressure_anomaly

    @inbounds pₕ′[i, j, 1] = anelastic_buoyancy(i, j, 1, grid, ref_state, temperature, specific_humidity, thermo)

    for k in 2:grid.Nz
        Δp′ = ℑzᵃᵃᶠ(i, j, k, grid, anelastic_buoyancy, ref_state, temperature, specific_humidity, thermo)
        @inbounds pₕ′[i, j, k] = pₕ′[i, j, k-1] + Δp′
    end
end

function update_hydrostatic_pressure!(model)
    grid = model.grid
    arch = grid.architecture
    pₕ′ = model.hydrostatic_pressure_anomaly
    ref_state = model.reference_state
    temperature = model.temperature
    specific_humidity = model.specific_humidity
    thermo = model.thermodynamics
    launch!(arch, grid, :xy, _update_hydrostatic_pressure!, pₕ′, grid, ref_state, temperature, specific_humidity, thermo)
    return nothing
end


function condensate_specific_humidity(T, state::AnelasticParcelState, thermo)
    qᵛ★ = saturation_specific_humidity(T, state.ρʳ, thermo, thermo.condensation)
    return max(0, state.q - qᵛ★)
end

@inline function anelastic_temperature(state::AnelasticParcelState{FT}, thermo) where FT
    state.θ == 0 && return zero(FT)

    # Generate guess for unsaturated conditions
    Π = state.Π
    T₁ = Π * state.θ
    qˡ₁ = condensate_specific_humidity(T₁, state, thermo)
    qˡ₁ <= 0 && return T₁
    
    # If we made it this far, we have condensation
    r₁ = saturation_adjustment_residual(T₁, Π, qˡ₁, state, thermo)

    ℒ = thermo.condensation.latent_heat
    cᵖᵐ = mixture_heat_capacity(state.q, thermo)
    T₂ = (T₁ + sqrt(T₁^2 + 4 * ℒ * qˡ₁ / cᵖᵐ)) / 2
    qˡ₂ = condensate_specific_humidity(T₂, state, thermo)
    r₂ = saturation_adjustment_residual(T₂, Π, qˡ₂, state, thermo)

    # Saturation adjustment
    R = sqrt(max(T₂, T₁))
    ϵ = convert(FT, 1e-4)
    δ = ϵ * R 
    iter = 0

    while abs(r₂ - r₁) > δ
        # Compute slope
        ΔTΔr = (T₂ - T₁) / (r₂ - r₁)

        # Store previous values
        r₁ = r₂
        T₁ = T₂

        # Update
        T₂ -= r₂ * ΔTΔr
        qˡ₂ = condensate_specific_humidity(T₂, state, thermo)
        r₂ = saturation_adjustment_residual(T₂, Π, qˡ₂, state, thermo)
        iter += 1
    end

    return T₂
end

@inline function saturation_adjustment_residual(T, Π, qˡ, state, thermo)
    ℒᵛ = thermo.condensation.latent_heat
    cᵖᵐ = mixture_heat_capacity(state.q, thermo)
    return T^2 - ℒᵛ * qˡ / cᵖᵐ - Π * state.θ * T
end

@inline function specific_volume(state, ref, thermo)
    T = temperature(state, ref, thermo)
    Rᵐ = mixture_gas_constant(state.q, thermo)
    pʳ = reference_pressure(state.z, ref, thermo)
    return Rᵐ * T / pʳ
end