module reactor_flux

using Interpolations
using DataStructures
using PCHIPInterpolation
using CSV
using DataFrames
using ..Newtrinos

export ReactorFluxConfig, HuberFlux

const datadir = @__DIR__


abstract type NominalFluxModel end


abstract type FluxSystematicsModel end


@kwdef struct ReactorFluxConfig{F<:NominalFluxModel, S<:FluxSystematicsModel}
    isotope_ratio::NamedTuple = (U_235 = 0.61, Pu_239 = 0.33, Pu_241 = 0.06,)
    nominal_model::F = HuberFlux(isotope_ratio)
    systematics_model::S = HuberSystematics()
end


struct HuberFlux <: NominalFluxModel
    isotope_ratio::NamedTuple
end

struct HuberSystematics <: FluxSystematicsModel
end


struct DayaBayFlux <: NominalFluxModel
    cfg::ReactorFluxConfig
end


struct DayaBaySystematics <: FluxSystematicsModel
end



@kwdef struct ReactorFlux <: Newtrinos.Physics
    cfg::ReactorFluxConfig
    params::NamedTuple
    priors::NamedTuple
    nominal_flux::Function
    sys_flux::Function
end

function configure(cfg::ReactorFluxConfig = ReactorFluxConfig())
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
    isotope_ratio = cfg.isotope_ratio
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

function get_sys_flux(cfg::HuberFlux)
    func(x) = 0.
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
    params = (;)
end

function get_priors(cfg::DayaBaySystematics)
    priors = (;)
end

function get_daya_bay_flux()
    # read in spectrum
    spectrum = CSV.read(joinpath(@__DIR__, "daya_bay_flux.csv"), DataFrame, header=1, delim="  ");
    bin_edges = sort(union(spectrum[!, 1], spectrum[!, 2]))
    flux = spectrum[!, 3]
    flux_errs = spectrum[!, 4]
    binc = bin_edges[1:end-1] + diff(bin_edges) / 2
    # read in correlation matrix
    correlation = CSV.read("daya_bay_flux_correlation.csv", DataFrame, delim="  ", header=0)
    correlation = Matrix(correlation[51:end, 51:end-1]);
    # decompose into eigenvectors
    decomp = eigen(correlation);
    diag = Diagonal(decomp.values);
    #convert eigenvalues (variances) to standard deviations
    stds = sqrt.(decomp.values);
    stacked_eigenvalues = reshape(repeat(stds, outer=stds.size), (25, 25));
    pull_matrix = decomp.vectors .* transpose(stacked_eigenvalues)

    # Create matrix
    M = zeros(length(binc), length(binc))   # first index i, second index n
    for (i, (l, h)) in enumerate(zip(bin_edges[1:end-1], bin_edges[2:end]))
        for n in eachindex(binc)
            func(E) = node_itp[n](E) * huber(E, (U_235=1,)) * x_sec_itp(E)
            integrand(E, p) = func(E) > 0 ? func(E) : 0.
            prob = IntegralProblem(integrand, (l, h))
            sol = solve(prob, QuadGKJL())
            M[i, n] = sol.u / (h - l)
        end
    end
    M_inv = inv(M)
    nodes = Diagonal(ones(25));
    node_itp = [Interpolator(binc, nodes[i, :]) for i in eachindex(binc)];
    y_0 = M_inv * flux
    y_nk = M_inv * pull_matrix;
    (;y_0, y_nk, node_itp)
end

function get_nominal_flux(cfg::DayaBayFlux)
    nominal_flux = get_daya_bay_flux()
    y_0 = nominal_flux.y_0
    y_nk = nominal_flux.y_nk
    node_itp = nominal_flux.node_itp
    
    function nominal(E)
        _huber = huber(E, cfg.isotope_Ratio)
        phi_0 = sum([y_0[i] * node_itp[i](E) for i in eachindex(y_0)])
        psi_k = sum([y_nk[n, :] * node_itp[n](E) for n in eachindex(y_0)])
        return _huber * phi_0
    end
    nominal
end

function get_sys_flux(cfg::DayaBayFlux)
    nominal_flux = get_daya_bay_flux()
    y_0 = nominal_flux.y_0
    y_nk = nominal_flux.y_nk
    node_itp = nominal_flux.node_itp
    
    function systematic(E, pulls)
        _huber = huber(E, cfg.isotope_Ratio)
        psi_k = sum([y_nk[n, :] * node_itp[n](E) for n in eachindex(y_0)])
        return _huber * sum(pulls .*psi_k)
    end
    systematic
end

end