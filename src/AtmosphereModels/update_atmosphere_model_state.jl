import Oceananigans.TimeSteppers: update_state!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ
using Oceananigans.ImmersedBoundaries: PartialCellBottom, ImmersedBoundaryGrid
using Oceananigans.Grids: topology
using Oceananigans.Grids: XFlatGrid, YFlatGrid

"""
    update_state!(model::AtmosphereModel, callbacks=[])

Update peripheral aspects of the model (halo regions, hydrostatic
pressure) to the current model state. If `callbacks` are provided (in an array),
they are called in the end.
"""
function update_state!(model::AtmosphereModel, callbacks=[]; compute_tendencies = true)

    # Mask immersed tracers
    foreach(model.tracers) do tracer
        mask_immersed_field!(tracer)
    end

    # Fill halos for velocities and tracers
    model_fields = fields(model)
    fill_halo_regions!(model_fields, model.clock, model_fields;
                       fill_boundary_normal_velocities = false, async = true)

    for callback in callbacks
        callback.callsite isa UpdateStateCallsite && callback(model)
    end

    # update_hydrostatic_pressure!(model)
    compute_tendencies && compute_tendencies!(model, callbacks)

    return nothing
end

function compute_tendencies!(model::NonhydrostaticModel, callbacks)

    grid = model.grid
    arch = grid.architecture

    kernel_parameters = :xyz
    compute_interior_tendency_contributions!(model, kernel_parameters; active_cells_map)

    for callback in callbacks
        callback.callsite isa TendencyCallsite && callback(model)
    end

    return nothing
end

""" Store previous value of the source term and compute current source term. """
function compute_interior_tendency_contributions!(model, kernel_parameters; active_cells_map = nothing)

    tendencies           = model.timestepper.Gⁿ
    arch                 = model.architecture
    grid                 = model.grid
    advection            = model.advection
    coriolis             = model.coriolis
    buoyancy             = model.buoyancy
    biogeochemistry      = model.biogeochemistry
    stokes_drift         = model.stokes_drift
    closure              = model.closure
    background_fields    = model.background_fields
    velocities           = model.velocities
    tracers              = model.tracers
    auxiliary_fields     = model.auxiliary_fields
    hydrostatic_pressure = model.pressures.pHY′
    diffusivities        = model.diffusivity_fields
    forcings             = model.forcing
    clock                = model.clock
    u_immersed_bc        = velocities.u.boundary_conditions.immersed
    v_immersed_bc        = velocities.v.boundary_conditions.immersed
    w_immersed_bc        = velocities.w.boundary_conditions.immersed

    start_momentum_kernel_args = (advection,
                                  coriolis,
                                  stokes_drift,
                                  closure)

    end_momentum_kernel_args = (buoyancy,
                                background_fields,
                                velocities,
                                tracers,
                                auxiliary_fields,
                                diffusivities)

    u_kernel_args = tuple(start_momentum_kernel_args...,
                          u_immersed_bc, end_momentum_kernel_args...,
                          hydrostatic_pressure, clock, forcings.u)

    v_kernel_args = tuple(start_momentum_kernel_args...,
                          v_immersed_bc, end_momentum_kernel_args...,
                          hydrostatic_pressure, clock, forcings.v)

    w_kernel_args = tuple(start_momentum_kernel_args...,
                          w_immersed_bc, end_momentum_kernel_args...,
                          hydrostatic_pressure, clock, forcings.w)

    exclude_periphery = true
    launch!(arch, grid, kernel_parameters, compute_Gu!, 
            tendencies.u, grid, u_kernel_args;
            active_cells_map, exclude_periphery)

    launch!(arch, grid, kernel_parameters, compute_Gv!, 
            tendencies.v, grid, v_kernel_args;
            active_cells_map, exclude_periphery)

    launch!(arch, grid, kernel_parameters, compute_Gw!, 
            tendencies.w, grid, w_kernel_args;
            active_cells_map, exclude_periphery)

    start_tracer_kernel_args = (advection, closure)
    end_tracer_kernel_args   = (buoyancy, biogeochemistry, background_fields, velocities,
                                tracers, auxiliary_fields, diffusivities)

    for tracer_index in 1:length(tracers)
        @inbounds c_tendency = tendencies[tracer_index + 3]
        @inbounds forcing = forcings[tracer_index + 3]
        @inbounds c_immersed_bc = tracers[tracer_index].boundary_conditions.immersed
        @inbounds tracer_name = keys(tracers)[tracer_index]

        args = tuple(Val(tracer_index), Val(tracer_name),
                     start_tracer_kernel_args...,
                     c_immersed_bc,
                     end_tracer_kernel_args...,
                     clock, forcing)

        launch!(arch, grid, kernel_parameters, compute_Gc!, 
                c_tendency, grid, args;
                active_cells_map)
    end

    return nothing
end

#####
##### Tendency calculators for u, v, w-velocity
#####

""" Calculate the right-hand-side of the u-velocity equation. """
@kernel function compute_Gu!(Gu, grid, args) 
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = u_velocity_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function compute_Gv!(Gv, grid, args) 
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = v_velocity_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the w-velocity equation. """
@kernel function compute_Gw!(Gw, grid, args) 
    i, j, k = @index(Global, NTuple)
    @inbounds Gw[i, j, k] = w_velocity_tendency(i, j, k, grid, args...)
end


#####
##### Tracer(s)
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function compute_Gc!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = tracer_tendency(i, j, k, grid, args...)
end

#####
##### Boundary contributions to tendencies due to user-prescribed fluxes
#####

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function compute_boundary_tendency_contributions!(Gⁿ, arch, velocities, tracers, clock, model_fields)
    fields = merge(velocities, tracers)

    foreach(i -> apply_x_bcs!(Gⁿ[i], fields[i], arch, clock, model_fields), 1:length(fields))
    foreach(i -> apply_y_bcs!(Gⁿ[i], fields[i], arch, clock, model_fields), 1:length(fields))
    foreach(i -> apply_z_bcs!(Gⁿ[i], fields[i], arch, clock, model_fields), 1:length(fields))

    return nothing
end



#=
@kernel function _update_hydrostatic_pressure!(pʰ, grid, buoyancy, C)
    i, j = @index(Global, NTuple)

    @inbounds pʰ[i, j, grid.Nz] = - z_dot_g_bᶜᶜᶠ(i, j, grid.Nz+1, grid, buoyancy, C) * Δzᶜᶜᶠ(i, j, grid.Nz+1, grid)

    for k in grid.Nz-1 : -1 : 1
        @inbounds pʰ[i, j, k] = pʰ[i, j, k+1] - z_dot_g_bᶜᶜᶠ(i, j, k+1, grid, buoyancy, C) * Δzᶜᶜᶠ(i, j, k+1, grid)
    end
end

function update_hydrostatic_pressure!(model; kwargs...)
    grid = model.grid
    arch = model.architecture
    pʰ = model.hydrostatic_pressure_anomaly
    parameters = hydrostatic_pressure_kernel_parameters(grid)
    launch!(arch, grid, parameters, _update_hydrostatic_pressure!, pʰ, grid, model.buoyancy, model.tracers)
end

# extend p kernel to compute also the boundaries
@inline function hydrostatic_pressure_kernel_parameters(grid)
    Nx, Ny, _ = size(grid)
    TX, TY, _ = topology(grid)

    ii = ifelse(TX == Flat, 1:Nx, 0:Nx+1)
    jj = ifelse(TY == Flat, 1:Ny, 0:Ny+1)

    return KernelParameters(ii, jj)
end
=#