module reactor_flux

using Interpolations
using DataStructures
using PCHIPInterpolation
using LinearAlgebra
using Integrals
using CSV
using DataFrames
using Distributions
using HDF5
using ..Newtrinos

export ReactorFluxConfig, HuberFlux, HuberSystematics, DayaBayFlux, DayaBaySystematics

const datadir = @__DIR__


abstract type NominalFluxModel end


abstract type FluxSystematicsModel end


# TODO: add name to flux config to identify reactors
@kwdef struct ReactorFluxConfig{F<:NominalFluxModel, S<:FluxSystematicsModel}#, X:<Newtrinos.ibd_xsec.IBDXsec}
    nominal_model::F = DayaBayFlux()
    systematics_model::S = DayaBaySystematics()
    isotope_ratio::NamedTuple = (
        U_235 = 0.563452,
        U_238 = 0.07593,
        Pu_239 = 0.304785,
        Pu_241 = 0.055834
    )
end


struct HuberFlux <: NominalFluxModel
end

struct HuberSystematics <: FluxSystematicsModel
end


struct DayaBayFlux <: NominalFluxModel
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
    # read in the hdf5 here, create flux taking the isotopes fractions as mixture coefficients
    path = "/home/iwsatlas1/kuhlmann/DEMOS/neutrinos/osc/dayabay_data/reactor_antineutrino_spectra_hm.hdf5"
    h5 = h5open(path, "r")
    E = [v.E_MeV for v in h5["Pu239"][:]]
    Pu_239 = [v.N_density for v in h5["Pu239"][:]]
    Pu_241 = [v.N_density for v in h5["Pu241"][:]]
    U_235 = [v.N_density for v in h5["U235"][:]]
    U_238 = [v.N_density for v in h5["U238"][:]]
    close(h5)


    f_239_interp = Interpolator(E, Pu_239)
    Pu_239(E) = isnan(f_239_interp(E)) ? 0.0 : f_239_interp(E)

    f_241_interp = Interpolator(E, Pu_241)
    Pu_241(E) = isnan(f_241_interp(E)) ? 0.0 : f_241_interp(E)

    f_238_interp = Interpolator(E, U_238)
    U_238(E) = isnan(f_238_interp(E)) ? 0.0 : f_238_interp(E)

    f_235_interp = Interpolator(E, U_235)
    U_235(E) = isnan(f_235_interp(E)) ? 0.0 : f_235_interp(E)

    names = (:U235, :U238, :Pu239, :Pu241)
    funcs = [U_235, U_238, Pu_239, Pu_241]

    fluxes = NamedTuple{names}(funcs)

    function nominal_flux(E, isotopes::NamedTuple)
        flux = sum([fluxes[iso](E) for (iso, weight) in pairs(isotopes)])
        return flux
    end

    return nominal_flux
end


function get_sys_flux(cfg::HuberSystematics)
    func(x) = 0.
end


function get_params(cfg::DayaBayFlux)
    params = (
        pulls = zeros(25),
    )
    return params
end

function get_priors(cfg::DayaBayFlux)
    exp = zeros(25)
    cv = Diagonal(ones(25))
    priors = (
        pulls = Distributions.MvNormal(exp, cv),
    )
    return priors
end

function get_daya_bay_flux()
    # read in spectrum
    spectrum = CSV.read(joinpath(@__DIR__, "daya_bay_flux.csv"), DataFrame, header=1, delim="  ");
    bin_edges = sort(union(spectrum[!, 1], spectrum[!, 2]))
    flux = spectrum[!, 3]
    flux_errs = spectrum[!, 4]
    binc = bin_edges[1:end-1] + diff(bin_edges) / 2
    # read in correlation matrix
    correlation = CSV.read(joinpath(@__DIR__, "daya_bay_flux_correlation.csv"), DataFrame, delim="  ", header=0)
    correlation = Matrix(correlation[51:end, 51:end-1])
    # decompose into eigenvectors
    decomp = eigen(correlation)
    diag = Diagonal(decomp.values)
    #convert eigenvalues (variances) to standard deviations
    stds = sqrt.(decomp.values)
    stacked_eigenvalues = reshape(repeat(stds, outer=stds.size), (25, 25))
    pull_matrix = decomp.vectors .* transpose(stacked_eigenvalues)
    nodes = Diagonal(ones(25))
    node_itp = []
    for i in eachindex(binc)
        interp = Interpolator(binc, nodes[i, :])
        func(E) = interp(E) > 0 ? interp(E) : 0.
        push!(node_itp, func)
    end
    
    isotope_ratio = (
        U235 = 0.563452,
        U238 = 0.07593,
        Pu239 = 0.304785,
        Pu241 = 0.055834
    )
    huber_config = ReactorFluxConfig(nominal_model=HuberFlux(), systematics_model=HuberSystematics(), isotope_ratio=isotope_ratio)
    huber_flux = configure(huber_config)
    huber = huber_flux.nominal_flux
    
    xsec = Newtrinos.ibd_xsec.configure()
    x_sec_itp = xsec.xsec

    x_sec_integral = []
    #integrate xsec over energy bins
    for (i, (l, h)) in enumerate(zip(bin_edges[1:end-1], bin_edges[2:end]))
        integrand(E, p) = x_sec_itp(E)
        prob = IntegralProblem(integrand, (l, h))
        sol = solve(prob, QuadGKJL())
        push!(x_sec_integral, sol.u)
    end

    # Create matrix
    M = zeros(length(binc), length(binc))   # first index i, second index n
    for (i, (l, h)) in enumerate(zip(bin_edges[1:end-1], bin_edges[2:end]))
        for n in eachindex(binc)
            func(E) = node_itp[n](E) * huber(E, isotope_ratio) * x_sec_itp(E)
            integrand(E, p) = func(E) > 0. ? func(E) : 0.
            prob = IntegralProblem(integrand, (l, h))
            sol = solve(prob, QuadGKJL())
            M[i, n] = sol.u / (h - l)
        end
    end
    M_inv = inv(M)
    y_0 = M_inv * flux
    y_nk = M_inv * pull_matrix;
    (;y_0, y_nk, node_itp)
end

function get_nominal_flux(cfg::DayaBayFlux)
    nominal_flux = get_daya_bay_flux()
    y_0 = nominal_flux.y_0
    y_nk = nominal_flux.y_nk
    node_itp = nominal_flux.node_itp
    isotope_ratio = (
        U235 = 0.563452,
        U238 = 0.07593,
        Pu239 = 0.304785,
        Pu241 = 0.055834
    )
    huber_config = ReactorFluxConfig(nominal_model=HuberFlux(), systematics_model=HuberSystematics(), isotope_ratio=isotope_ratio)
    flux = configure(huber_config)
    huber = flux.nominal_flux
    function nominal(E)
        _huber = huber(E, isotope_ratio)
        phi_0 = sum([y_0[i] * node_itp[i](E) for i in eachindex(y_0)])
        #psi_k = sum([y_nk[n, :] * node_itp[n](E) for n in eachindex(y_0)])
        return _huber * phi_0
    end
    nominal
end

function get_sys_flux(cfg::DayaBaySystematics)
    isotope_ratio = (
        U235 = 0.563452,
        U238 = 0.07593,
        Pu239 = 0.304785,
        Pu241 = 0.055834
    )
    nominal_flux = get_daya_bay_flux()
    y_0 = nominal_flux.y_0
    y_nk = nominal_flux.y_nk
    node_itp = nominal_flux.node_itp
    huber_config = ReactorFluxConfig(nominal_model=HuberFlux(), systematics_model=HuberSystematics(), isotope_ratio=isotope_ratio)
    flux = configure(huber_config)
    huber = flux.nominal_flux

    
    function systematic(E, pulls)
        _huber = huber(E, isotope_ratio)
        psi_k = sum([y_nk[n, :] .* node_itp[n](E) for n in eachindex(y_0)])
        return _huber * sum(pulls .* psi_k)
    end
    systematic
end

end