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

export ReactorFluxConfig, DayaBayFlux


const datadir = joinpath(@__DIR__, "../experiments/dayabay/daya_bay_3158days")


abstract type FluxModel end


@kwdef struct ReactorFluxConfig{F<:FluxModel}
    flux_model::F = DayaBayFlux()
end


struct DayaBayFlux <: FluxModel
end

@kwdef struct ReactorFlux <: Newtrinos.Physics
    cfg::ReactorFluxConfig
    params::NamedTuple
    priors::NamedTuple
    flux::Function
    datadir::String
end


function configure(cfg::ReactorFluxConfig = ReactorFluxConfig())
    ReactorFlux(
        cfg=cfg,
        params = get_params(cfg.flux_model, datadir=cfg.datadir),
        priors = get_priors(cfg.flux_model, datadir=cfg.datadir),
        flux = get_flux(cfg.flux_model, datadir=cfg.datadir),
    )
end


function get_params(flux::DayaBayFlux; datadir = datadir)

    ## energy per fission
    file = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_energy_per_fission.yaml"))
    data = file["parameters"]["energy_per_fission"]

    energy_per_fission = [data["U235"][1], data["U238"][1], data["Pu239"][1], data["Pu241"][1]]


    ## reactor thermal power
    data = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_thermal_power_uncertainty.yaml"))
    reactor_thermal_power_scale = data["parameters"]["thermal_power_scale"][1] .* ones(6)

    ## fission fraction scales
    data = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_fission_fractions_scale.yaml"), dicttype=OrderedDict{String,Any})
    fraction_scale = data["parameters"]["fission_fractions_scale"]

    names = data["correlations"]["fission_fractions_scale"]["names"]
    fission_fractions_scale = collect([Float64(fraction_scale[name][1]) for name in names])

    params = (;
        energy_per_fission,
        reactor_thermal_power_scale,
        fission_fractions_scale,
    )
    return params
end


function get_priors(flux::DayaBayFlux; datadir = datadir)

    ## energy per fission
    file = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_energy_per_fission.yaml"))
    data = file["parameters"]["energy_per_fission"]

    mu = [data["U235"][1], data["U238"][1], data["Pu239"][1], data["Pu241"][1]]
    var = [data["U235"][2], data["U238"][2], data["Pu239"][2], data["Pu241"][2]].^2
    energy_per_fission = Distributions.MvNormal(mu, Diagonal(var))
    #energy_per_fission_U235 = Distributions.Normal(data["U235"][1], data["U235"][1])
    #energy_per_fission_U238 = Distributions.Normal(data["U238"][1], data["U238"][1])
    #energy_per_fission_Pu239 = Distributions.Normal(data["Pu239"][1], data["Pu239"][1])
    #energy_per_fission_Pu241 = Distributions.Normal(data["Pu241"][1], data["Pu241"][1])

    #energy_per_fission = (
    #    U235=energy_per_fission_U235,
    #    U238=energy_per_fission_U238,
    #    Pu239=energy_per_fission_Pu239,
    #    Pu241=energy_per_fission_Pu241,
    #    )


    ## reactor thermal power uncertainty
    data = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_thermal_power_uncertainty.yaml"))
    mu = data["parameters"]["thermal_power_scale"][1]
    sigma = data["parameters"]["thermal_power_scale"][2] * mu * 0.01  # percent
    reactor_thermal_power_scale = Distributions.MvNormal(mu .* ones(6), Diagonal(sigma.^2 .* ones(6)))

    ## fission fraction scale

    data = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_fission_fractions_scale.yaml"), dicttype=OrderedDict{String,Any})
    fraction_scale = data["parameters"]["fission_fractions_scale"]

    isotopes = ["U235", "U238", "Pu239", "Pu241"]
    scale = zeros((4, 4))
    for i in 1:4
        for j in 1:4
            scale[i, j] = fraction_scale[isotopes[i]][1] * fraction_scale[isotopes[i]][2] * fraction_scale[isotopes[j]][1] * fraction_scale[isotopes[j]][2] * 0.01 * 0.01   # percent
        end
    end

    mu = collect([Float64(fraction_scale[name][1]) for name in isotopes])
    corr_mat = hcat(data["correlations"]["fission_fractions_scale"]["matrix"]...)


    cov_mat = corr_mat .* scale
    fission_fractions_scale = Distributions.MvNormal(mu, cov_mat)

    priors = (;
        energy_per_fission,
        reactor_thermal_power_scale,
        fission_fractions_scale,
    )

end


function extract_reactor_spectra(datadir = datadir)
    file = h5open(joinpath(datadir, "dayabay_data/reactor_antineutrino_spectra_hm.hdf5"))
    spec_Pu239 = Float64[]
    spec_Pu241 = Float64[]
    spec_U235 = Float64[]
    spec_U238 = Float64[]

    E = Float64[]

    foreach(x -> (push!(spec_Pu239, x[2]), push!(E, x[1])), file["Pu239"][1:end])
    foreach(x -> push!(spec_Pu241, x[2]), file["Pu241"][1:end])
    foreach(x -> push!(spec_U235, x[2]), file["U235"][1:end])
    foreach(x -> push!(spec_U238, x[2]), file["U238"][1:end])

    # TODO uncertainties
    corr_Pu239 = Float64[]
    corr_Pu241 = Float64[]
    corr_U238 = Float64[]
    corr_U235 = Float64[]
    uncorr_Pu239 = Float64[]
    uncorr_Pu241 = Float64[]
    uncorr_U238 = Float64[]
    uncorr_U235 = Float64[]

    return (Pu239=spec_Pu239, Pu241=spec_Pu241, U238=spec_U238, U235=spec_U235, E=E)
end



function get_flux(cfg::DayaBayFlux; datadir = datadir)

    # get fission fractions for fixed weighting
    fractions = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_fission_fractions.yaml"))["parameters"]["fission_fractions"]

    fractions = NamedTuple((Symbol(key),value) for (key,value) in fractions)

    fluxes = extract_reactor_spectra(datadir)
    E = fluxes.E

    f_239_interp = Interpolator(E, fluxes.Pu239)
    Pu_239(E) = isnan(f_239_interp(E)) ? 0.0 : f_239_interp(E)

    f_241_interp = Interpolator(E, fluxes.Pu241)
    Pu_241(E) = isnan(f_241_interp(E)) ? 0.0 : f_241_interp(E)

    f_238_interp = Interpolator(E, fluxes.U238)
    U_238(E) = isnan(f_238_interp(E)) ? 0.0 : f_238_interp(E)

    f_235_interp = Interpolator(E, fluxes.U235)
    U_235(E) = isnan(f_235_interp(E)) ? 0.0 : f_235_interp(E)

    names = (:U235, :U238, :Pu239, :Pu241)
    funcs = [U_235, U_238, Pu_239, Pu_241]
    fluxes = NamedTuple{names}(funcs)

    # get nominal thermal power
    file = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/reactor_thermal_power_nominal.yaml"))
    nominal_thermal_power = file["parameters"]["nominal_thermal_power"]   # GW

    # neutrinos per fission, is fixed
    file = YAML.load_file(joinpath(datadir, "dayabay_data/parameters/antineutrinos_per_fission_huber_mueller.yaml"))
    nu_per_f = file["parameters"]["antineutrinos_per_fission"]

    nu_per_fission = NamedTuple((Symbol(key), value) for (key, value) in nu_per_f)


    elementary_charge = 1.602176634e-19

    GJ_to_MeV = 1e9 / elementary_charge * 1e-6 # reactor thermal power in GW, energy per fission in MeV
    # all spectra in MeV, hence get rid of the GJ

    function reactor_flux(E, thermal_power_scale, energy_per_fission, fission_fractions_scale)
        
        flux = GJ_to_MeV * thermal_power_scale * nominal_thermal_power * sum([
            nu_per_fission[iso] * fission_fractions_scale[i] * fractions[iso] * fluxes[iso].(E) for (i, iso) in enumerate(names)]
            ) / sum(
                [fission_fractions_scale[i]  * energy_per_fission[i] for (i, iso) in enumerate(names)]
            )
    end

    reactor_flux
end


end