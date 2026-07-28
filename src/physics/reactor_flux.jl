module reactor_flux

using Interpolations
using DataStructures
using PCHIPInterpolator
using CSV
using DataFrames
using ..Newtrinos

export ReactorFluxConfig, DayaBay, DayaBaySystematics

const datadir = @__DIR__


abstract type NominalFlux end

struct HuberFlux <: NominalFluxModel
    cfg::ReactorFluxConfig
end

struct DayaBayFlux <: NominalFluxModel
    cfg::ReactorFluxConfig
end

abstract type FluxSystematicsModel end

struct DayaBaySystematics <: FluxSystematicsModel
end


@kwdef struct ReactorFluxConfig{F<:NominalFluxModel, S<:FluxSystematicsModel}
    nominal_model::F = DayaBayFlux()
    systematics_model::S = DayaBaySystematics()
    isotope_ratio::NamedTuple = (U_235 = 1., U_239 = 0., Pu_241 = 0.,)
end

@kwdef struct ReactorFlux <: Newtrinos.Physics
    cfg::ReactorFluxConfig
    params::NamedTuple
    priors::NamedTuple
    nominal_flux::Function
    sys_flux::Function
end

function configure(cfg::ReactorFluxConfig=ReactorFluxConfig())
    ReactorFlux(
        cfg=cfg,
        params = get_params(cfg.nominal_model),
        priors = get_priors(cfg.nominal_model),
        nominal_flux = get_nominal_flux(cfg.nominal_model),
        sys_flux = get_sys_flux(cfg.systematics_model)
    )
end


function get_params(cfg::HuberFlux)
    return (;)
end

function get_priors(cfg::HuberFlux)
    return (;)
end

function get_nominal_flux(cfg::HuberFlux)
    isotope_ratio = cfg.cfg.isotope_ratio
    U_235 = [4.367, -4.577, 2.100, -5.294e-1, 6.186e-2, -2.777e-3]
    Pu_239 = [4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3]
    Pu_241 = [2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3]
    coeffs = (;U_235, Pu_239, Pu_241)
    
    function nominal_flux(E)
        function single_flux(E, element)
            coeff = coeffs[element]
            exponent = sum([coeff[i] * E^(i-1) for i in eachindex(coeff)])
            return exp(exponent)
        end
        isotope_weighted_flux = 0.
        for (iso, weight) in pairs(fraction)
            isotope_weighted_flux += weight * single_flux(E, iso)
        end
        return isotope_weighted_flux

    end

    nominal_flux
end

function get_priors(cfg::HuberFlux)
    return (;)
end

function get_params(cfg::HuberFlux)
    return (;)
end

"""
function configure(cfg::ReactorFluxConfig=ReactorFluxConfig())
    ReactorFlux(
        cfg=cfg,
        params = get_params(cfg.systematics_model),
        priors = get_priors(cfg.systematics_model),
        nominal_flux = get_nominal_flux(cfg.nominal_model),
        sys_flux = get_sys_flux(cfg.systematics_model)
        )
end
"""

function get_params(cfg::DayaBaySystematics)
    params = ()
end

function get_priors(cfg::DayaBaySystematics)
    priors = ()
end

function get_daya_bay_flux(filename)
    spectrum = CSV.read(joinpath(@__DIR__, "daya_bay_flux.csv"), DataFrame, header=1, delim="  ");
    bin_edges = sort(union(spectrum[!, 1], spectrum[!, 2]))
    flux = spectrum[!, 3]
    flux_errs = spectrum[!, 4]
    binc = bin_edges[1:end-1] + diff(bin_edges) / 2
    itp = Interpolator(binc, flux)
end

function get_nominal_flux(cfg::DayaBay)
end
    
function get_nominal_flux(cfg::DayaBayFlux)
end

end