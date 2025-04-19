struct IdealGas{FT}
    molar_mass :: FT
    isentropic_exponent :: FT # κ = γ / (γ - 1)
end

function IdealGas(FT = Oceananigans.defaults.FloatType;
                  molar_mass = 0.02897,
                  isentropic_exponent = 2/7)

    return IdealGas{FT}(convert(FT, molar_mass),
                        convert(FT, isentropic_exponent))
end

struct Condensation{FT}
    reference_vaporization_enthalpy :: FT
    energy_reference_temperature :: FT
    triple_point_temperature :: FT
    triple_point_pressure :: FT
    liquid_heat_capacity :: FT
end

function Condensation(FT = Oceananigans.defaults.FloatType;
                      reference_vaporization_enthalpy = 2500800,
                      energy_reference_temperature = 273.16,
                      vapor_heat_capacity = 1859,
                      triple_point_temperature = 273.16,
                      triple_point_pressure = 611.657)

    return Condensation{FT}(convert(FT, reference_vaporization_enthalpy),
                            convert(FT, energy_reference_temperature),
                            convert(FT, vapor_heat_capacity),
                            convert(FT, triple_point_temperature),
                            convert(FT, triple_point_pressure))
end

struct AtmosphereThermodynamics{FT, C}
    molar_gas_constant :: FT
    dry_air :: IdealGas{FT}
    vapor :: IdealGas{FT}
    condensation :: C
end

"""
    AtmosphereThermodynamics(FT = Oceananigans.defaults.FloatType;
                             molar_gas_constant = 8.314462618,
                             dry_air_molar_mass = 0.02897,
                             dry_air_isentropic_exponent = 2/7,
                             vapor_molar_mass = 0.018015,
                             vapor_isentropic_exponent = 4.03,
                             condensation = nothing)

Create `AtmosphereThermodynamics` with parameters that correpsond to the composition of dry air
in Earth's atmosphere and water vapor. The default `isnothing(condensation)` implies thermodynamics
appropriate for unsaturated and therefore non-condensing air.
"""
function AtmosphereThermodynamics(FT = Oceananigans.defaults.FloatType;
                                  molar_gas_constant = 8.314462618,
                                  dry_air_molar_mass = 0.02897,
                                  dry_air_isentropic_exponent = 2/7,
                                  vapor_molar_mass = 0.018015,
                                  vapor_isentropic_exponent = 4.03,
                                  condensation = nothing)

    dry_air = IdealGas(FT; molar_mass = dry_air_molar_mass,
                           isentropic_exponent = dry_air_isentropic_exponent)

    vapor = IdealGas(FT; molar_mass = vapor_molar_mass,
                         isentropic_exponent = vapor_isentropic_exponent)

    return AtmosphereThermodynamics(convert(FT, molar_gas_constant),
                                    dry_air, vapor, condensation)
end

#####
##### Computations
#####

@inline gas_heat_capacity(molar_gas_constant, igc::IdealGas)           = igc.isentropic_exponent * gas_constant / igc.molar_mass
@inline dry_air_heat_capacity(tc::AtmosphereThermodynamics)            = gas_heat_capacity(tc.molar_gas_constant, tc.dry_air)
@inline vapor_heat_capacity(tc::AtmosphereThermodynamics)              = gas_heat_capacity(tc.molar_gas_constant, tc.vapor)
@inline vapor_specific_gas_constant(tc::AtmosphereThermodynamics)      = tc.molar_gas_constant / tc.vapor.molar_mass
@inline dry_air_specific_gas_constant(tc::AtmosphereThermodynamics)    = tc.molar_gas_constant / tc.dry_air.molar_mass

const NonCondensingAtmosphereThermodynamics{FT} = AtmosphereThermodynamics{FT, Nothing} 

@inline function saturation_vapor_pressure(T, thermodynamics_constants)
    ℒ₀  = thermodynamics_constants.condesation.reference_vaporization_enthalpy
    T₀  = thermodynamics_constants.condesation.energy_reference_temperature
    Tᵗʳ = thermodynamics_constants.condesation.triple_point_temperature
    pᵗʳ = thermodynamics_constants.condesation.triple_point_pressure
    cᵖℓ = thermodynamics_constants.condesation.liquid_heat_capacity
    cᵖv = vapor_specific_heat_capacity(thermodynamics_constants)
    Rᵛ  = vapor_specific_gas_constant(thermodynamics_constants)
    Δc  = cᵖℓ - cᵖv

    return pᵗʳ * (T / Tᵗʳ)^(Δc / Rᵛ) * exp((ℒ₀ - Δc * T₀) * (1/Tᵗʳ - 1/T) / Rᵛ)
end

@inline function saturation_specific_humidity(T, ρ, thermodynamics_constants)
    p★ = saturation_vapor_pressure(T, thermodynamics_constants)
    Rᵛ = vapor_specific_gas_constant(thermodynamics_constants)
    return p★ / (ρ * Rᵛ * T)
end
