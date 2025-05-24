using Adapt

struct IdealGas{FT}
    molar_mass :: FT
    heat_capacity :: FT # specific heat capacity at constant pressure
end

Base.eltype(::IdealGas{FT}) where FT = FT

function Adapt.adapt_structure(to, gas::IdealGas)
    molar_mass = adapt(to, gas.molar_mass)
    heat_capacity = adapt(to, gas.heat_capacity)
    FT = typeof(molar_mass)
    return IdealGas{FT}(molar_mass, heat_capacity)
end

function IdealGas(FT = Oceananigans.defaults.FloatType;
                  molar_mass = 0.02897,
                  heat_capacity = 1005)

    return IdealGas{FT}(convert(FT, molar_mass),
                        convert(FT, heat_capacity))
end

struct Saturation{FT}
    energy_reference_temperature :: FT
    triple_point_temperature :: FT
    triple_point_pressure :: FT
end

Adapt.adapt_structure(to, sat::Saturation) =
    Saturation(adapt(to, sat.energy_reference_temperature),
               adapt(to, sat.triple_point_temperature),
               adapt(to, sat.triple_point_pressure))

"""
    Saturation(FT = Oceananigans.defaults.FloatType;
               energy_reference_temperature,
               triple_point_temperature,
               triple_point_pressure)

Returns `Saturation` with specified parameters converted to `FT`.

The `Saturation` struct contains reference values used in the Clausius-Clapeyron relation to calculate
saturation vapor pressure. The Clausius-Clapeyron equation describes how the saturation vapor pressure
changes with temperature for a liquid-vapor phase transition:

The Clausius-Clapeyron relation describes the pressure-temperature relationship during phase transitions:

```math
    d/dT log(p_sat) = L / (Rv * T²)
```
where:

- p_sat is the saturation vapor pressure
- T is temperature 
- L is the latent heat of vaporization
- R_v is the gas constant for water vapor

For water vapor, this integrates to:

    ln(p_sat/p_triple) = (L/R_v) * (1/T_triple - 1/T)

or equivalently:

    p_sat = p_triple * exp((L/R_v) * (1/T_triple - 1/T))

where:
- p_triple is the triple point pressure
- T_triple is the triple point temperature

This relation is used to calculate saturation vapor pressure, which determines the maximum possible
water vapor content of air at a given temperature and pressure.
"""
function Saturation(FT = Oceananigans.defaults.FloatType;
                    energy_reference_temperature = 273.16,
                    triple_point_temperature = 273.16,
                    triple_point_pressure = 611.657)

    return Saturation{FT}(convert(FT, energy_reference_temperature),
                          convert(FT, triple_point_temperature), 
                          convert(FT, triple_point_pressure))
end

struct PhaseTransition{FT}
    latent_heat :: FT
    heat_capacity :: FT
end

Adapt.adapt_structure(to, pt::PhaseTransition) =
    PhaseTransition(adapt(to, pt.latent_heat),
                    adapt(to, pt.heat_capacity))

"""
    PhaseTransition(FT = Oceananigans.defaults.FloatType; latent_heat, heat_capacity)

Returns `PhaseTransition` with specified parameters converted to `FT`.

Two examples of `PhaseTransition` are condensation and deposition.
During condensation, water molecules in the gas phase cluster together and slow down to form liquid with `heat_capacity`,
The lost of molecular kinetic energy is called the `latent_heat`.

Likewise, during deposition, water molecules in the gas phase cluster into ice crystals.

Arguments
=========
- `FT`: Float type to use (defaults to Oceananigans.defaults.FloatType)
- `latent_heat`: Latent heat lost during phase transition from the gaseous state.
- `heat_capacity`: Heat capacity of the transitional phase.
"""
function PhaseTransition(FT = Oceananigans.defaults.FloatType; latent_heat, heat_capacity)
    return PhaseTransition{FT}(convert(FT, latent_heat),
                               convert(FT, heat_capacity))
end

water_condensation(FT) = PhaseTransition(FT; latent_heat=2500800, heat_capacity=4181)
water_deposition(FT)   = PhaseTransition(FT; latent_heat=2834000, heat_capacity=2108)

struct AtmosphereThermodynamics{FT, S, C, F}
    molar_gas_constant :: FT
    gravitational_acceleration :: FT
    dry_air :: IdealGas{FT}
    vapor :: IdealGas{FT}
    saturation :: S
    condensation :: C
    deposition :: F
end

Base.eltype(::AtmosphereThermodynamics{FT}) where FT = FT

function Adapt.adapt_structure(to, thermo::AtmosphereThermodynamics)
    molar_gas_constant = adapt(to, thermo.molar_gas_constant)   
    gravitational_acceleration = adapt(to, thermo.gravitational_acceleration)
    dry_air = adapt(to, thermo.dry_air)
    vapor = adapt(to, thermo.vapor)
    saturation = adapt(to, thermo.saturation)
    condensation = adapt(to, thermo.condensation)
    deposition = adapt(to, thermo.deposition)
    FT = typeof(molar_gas_constant)
    S = typeof(saturation)
    C = typeof(condensation)
    F = typeof(deposition)
    return AtmosphereThermodynamics{FT, S, C, F}(molar_gas_constant,
                                                 gravitational_acceleration,
                                                 dry_air,
                                                 vapor,
                                                 saturation,
                                                 condensation,
                                                 deposition)
end

"""
    AtmosphereThermodynamics(FT = Oceananigans.defaults.FloatType;
                             gravitational_acceleration = 9.81,
                             molar_gas_constant = 8.314462618,
                             dry_air_molar_mass = 0.02897,
                             dry_air_heat_capacity = 1005,
                             vapor_molar_mass = 0.018015,
                             vapor_heat_capacity = 1850,
                             saturation = Saturation(FT),
                             condensation = water_condensation(FT),
                             deposition = water_deposition(FT))

Create `AtmosphereThermodynamics` with parameters that correpsond to the composition of dry air
in Earth's atmosphere and water vapor. The default `isnothing(condensation)` implies thermodynamics
appropriate for unsaturated and therefore non-condensing air.
"""
function AtmosphereThermodynamics(FT = Oceananigans.defaults.FloatType;
                                  gravitational_acceleration = 9.81,
                                  molar_gas_constant = 8.314462618,
                                  dry_air_molar_mass = 0.02897,
                                  dry_air_heat_capacity = 1005,
                                  vapor_molar_mass = 0.018015,
                                  vapor_heat_capacity = 1850,
                                  saturation = Saturation(FT),
                                  condensation = water_condensation(FT),
                                  deposition = water_deposition(FT))

    dry_air = IdealGas(FT; molar_mass = dry_air_molar_mass,
                           heat_capacity = dry_air_heat_capacity)

    vapor = IdealGas(FT; molar_mass = vapor_molar_mass,
                         heat_capacity = vapor_heat_capacity)

    return AtmosphereThermodynamics(convert(FT, molar_gas_constant),
                                    convert(FT, gravitational_acceleration),
                                    dry_air, vapor, saturation, condensation, deposition)
end

#####
##### Computations
#####

const AT = AtmosphereThermodynamics
const IG = IdealGas

@inline vapor_gas_constant(thermo::AT)    = thermo.molar_gas_constant / thermo.vapor.molar_mass
@inline dry_air_gas_constant(thermo::AT)  = thermo.molar_gas_constant / thermo.dry_air.molar_mass

const NonCondensingAtmosphereThermodynamics{FT} = AtmosphereThermodynamics{FT, Nothing} 

"""
    saturation_vapor_pressure(T, thermo)

Compute the saturation vapor pressure over a liquid surface by integrating
the Clausius-Clapeyron relation,

```math
dp/dT = ℒᵛ / (Rᵛ T^2)
```
"""
@inline function saturation_vapor_pressure(T, thermo, phase_transition::PhaseTransition)
    ℒ₀  = phase_transition.latent_heat
    cᵖˡ = phase_transition.heat_capacity
    T₀  = thermo.saturation.energy_reference_temperature
    Tᵗʳ = thermo.saturation.triple_point_temperature
    pᵗʳ = thermo.saturation.triple_point_pressure
    cᵖᵛ = thermo.vapor.heat_capacity
    Rᵛ  = vapor_gas_constant(thermo)

    Δcᵖ = cᵖˡ - cᵖᵛ
    Δϰ = Δcᵖ / Rᵛ

    return pᵗʳ * (T / Tᵗʳ)^Δϰ * exp((ℒ₀ - Δcᵖ * T₀) * (1/Tᵗʳ - 1/T) / Rᵛ)
end

# Over a liquid surface
@inline function saturation_specific_humidity(T, ρ, thermo, phase_transition::PhaseTransition)
    p★ = saturation_vapor_pressure(T, thermo, phase_transition)
    Rᵛ = vapor_gas_constant(thermo)
    return p★ / (ρ * Rᵛ * T)
end

@inline function mixture_gas_constant(q, thermo::AT)
    Rᵈ = dry_air_gas_constant(thermo)   
    Rᵛ = vapor_gas_constant(thermo)   
    return Rᵈ * (1 - q) + Rᵛ * q
end

"""
    mixture_heat_capacity(q, thermo)

Compute the heat capacity of state air given the total specific humidity q
and assuming that condensate mass ratio qᶜ ≪ q, where qℓ is the mass ratio of
liquid condensate.
"""
@inline function mixture_heat_capacity(q, thermo::AT)
    cᵖᵈ = thermo.dry_air.heat_capacity
    cᵖᵛ = thermo.vapor.heat_capacity
    return cᵖᵈ * (1 - q) + cᵖᵛ * q
end

#=
For example, if we would like to account for the mass of condensate consistently:
struct WarmCondensate{FT}
    vapor :: FT
    liquid :: FT
    total :: FT
end

# Then more correctly:
@inline function mixture_heat_capacity(q::WarmCondensate, thermo::AT)
    cᵖᵈ = dry_air_heat_capacity(thermo)
    cᵖᵛ = vapor_heat_capacity(thermo)
    cᵖˡ = thermo.condensation.liquid_heat_capacity
    qᵗ = q.total
    qᵛ = q.vapor
    qˡ = q.liquid
    return cᵖᵈ * (1 - qᵗ) + cᵖᵛ * qᵛ + cᵖˡ * qˡ
end

@inline function mixture_gas_constant(q::WarmCondensate, thermo::AT)
=#

#####
##### state thermodynamics for a Boussinesq model
#####

# Organizing information about the state is a WIP
struct ThermodynamicState{FT}
    θ :: FT
    q :: FT
    z :: FT
end

struct ReferenceState{FT}
    p₀ :: FT # base pressure: reference pressure at z=0
    θ :: FT  # constant reference potential temperature
end

Adapt.adapt_structure(to, ref::ReferenceState) =
    ReferenceState(adapt(to, ref.p₀),
                   adapt(to, ref.θ))

function ReferenceState(FT = Oceananigans.defaults.FloatType;
                        base_pressure = 101325,
                        potential_temperature = 288)

    return ReferenceState{FT}(convert(FT, base_pressure),
                              convert(FT, potential_temperature))
end

"""
    reference_density(ref, thermo)

Compute the reference density associated with the reference pressure and potential temperature.
The reference density is defined as the density of dry air at the reference pressure and temperature.
"""
@inline function reference_density(z, ref, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    p = reference_pressure(z, ref, thermo)
    return p / (Rᵈ * ref.θ)
end

@inline function base_density(ref, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    return ref.p₀ / (Rᵈ * ref.θ)
end

@inline function reference_specific_volume(z, ref, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    p = reference_pressure(z, ref, thermo)
    return Rᵈ * ref.θ / p
end

@inline function reference_pressure(z, ref, thermo)
    cᵖᵈ = thermo.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(thermo)
    ϰᵈ⁻¹ = Rᵈ / cᵖᵈ
    g = thermo.gravitational_acceleration
    return ref.p₀ * (1 - g * z / (cᵖᵈ * ref.θ))^ϰᵈ⁻¹
end

@inline function saturation_specific_humidity(T, z, ref::ReferenceState, thermo, phase_transition)
    ρ = reference_density(z, ref, thermo)
    return saturation_specific_humidity(T, ρ, thermo, phase_transition)
end

@inline function exner_function(state, ref, thermo)
    Rᵐ = mixture_gas_constant(state.q, thermo)
    cᵖᵐ = mixture_heat_capacity(state.q, thermo)
    inv_ϰᵐ = Rᵐ / cᵖᵐ
    pᵣ = reference_pressure(state.z, ref, thermo)
    return (pᵣ / ref.p₀)^inv_ϰᵐ
end

condensate_specific_humidity(T, state, ref, thermo) =
    condensate_specific_humidity(T, state.q, state.z, ref, thermo)

function condensate_specific_humidity(T, q, z, ref, thermo)
    qᵛ★ = saturation_specific_humidity(T, z, ref, thermo, thermo.condensation)
    return max(0, q - qᵛ★)
end

function ice_specific_humidity(T, q, z, ref, thermo)
    qi★ = saturation_specific_humidity(T, z, ref, thermo, thermo.deposition)
    return max(0, q - qi★)
end
