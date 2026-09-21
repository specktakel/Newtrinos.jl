---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: Julia 1.12
    language: julia
    name: julia-1.12
---

```julia
using Pkg
Pkg.activate("../../../..")
using Revise
using DataFrames
using CSV
using CairoMakie
using DelimitedFiles
using DataStructures
using LinearAlgebra
using Integrals
using PCHIPInterpolation
using Interpolations
using Distributions
using DensityInterface
using BAT
using DataStructures
using Newtrinos
using FileIO
using Accessors
using CairoMakie
using CSV, DataFrames
using ForwardDiff
using HDF5
using OrderedCollections
import YAML
using Newtrinos
```

```julia
str = "AD31"
str[3]
```

```julia
detector_list = []

function AD_to_EH(AD::String)
    return parse(Int, AD[3])
end


for p in period_list
    for AD in detector_dict[p]
        push!(detector_list, "$(p)$(AD)")
    end
end

detector_list
```

```julia
idx = findall(x-> x=="6ADAD12", detector_list)[1]
```

```julia
# make some global list of the detectors used in all the periods

dict = YAML.load_file("dayabay_data/parameters/background_rates_uncorrelated.yaml", dicttype=OrderedDict{String,Any})


period_list = ["6AD", "8AD", "7AD"]

EH_list = [1, 2, 3]

full_setup = ["AD11", "AD12", "AD21", "AD22", "AD31", "AD32", "AD33", "AD34"]
mask_6 =     [true,   true,   true,   false,  true,   true,   true,   false]
mask_8 =     [true,   true,   true,   true,   true,   true,   true,   true]
mask_7 =     [true,   false,  true,   true,   true,   true,   true,   true]

detectors_6AD = full_setup[mask_6]
detectors_8AD = full_setup[mask_8]
detectors_7AD = full_setup[mask_7]

detector_dict = Dict("6AD"=>detectors_6AD, "8AD"=>detectors_8AD, "7AD"=>detectors_7AD)

# detector_list = vcat(full_setup[mask_6], full_setup[mask_8], full_setup[mask_7])

period_EH_list = []
for p in period_list
    for EH in EH_list
        push!(period_EH_list, "$(p)_$(EH)")
    end
end


function get_effective_livetime(period::Int)
    daily_path = joinpath(data_basepath, "dayabay_daily_detector_data.hdf5")
    data = h5open(daily_path)
    eff_livetime = []
    livetime = []
    foreach(x -> x[:n_det] == period ? push!(eff_livetime, x[:eff_livetime]) : 0, data["AD$(AD)"][1:end])
    foreach(x -> x[:n_det] == period ? push!(livetime, x[:livetime]) : 0, data["AD$(AD)"][1:end])

    close(data)
    return sum(eff_livetime)
end


function extract_for_AD_period(AD::Int, period::Int)
    # extract IBD rate
    data_basepath = "dayabay_data/dayabay_dataset"
    spectrum_path = joinpath(data_basepath, "dayabay_ibd_spectra_$(period)AD.hdf5")
    file = h5open(spectrum_path, "r")

    counts_ibd = []
    E_bins_MeV = []

    foreach(x -> push!(counts_ibd, x.N), file["ibd_spectrum_AD$(AD)"][1:end])
    foreach(x -> push!(E_bins_MeV, x.E_min_MeV), file["ibd_spectrum_AD$(AD)"][1:end])
    push!(E_bins_MeV, file["ibd_spectrum_AD$(AD)"][end].E_max_MeV)
    close(file)
    E_center_MeV = (E_bins_MeV[1:end-1] .+ E_bins_MeV[2:end]) ./ 2

    # extract background shapes
    bg_shape_path = joinpath(data_basepath, "dayabay_background_spectra_$(period)AD.hdf5")
    file = h5open(bg_shape_path)
    shape_accidental = []
    shape_alpha_neutron = []
    shape_amc = []
    shape_fast_neutrons = []
    shape_lithium_helium = []

    foreach(x -> push!(shape_accidental, x.N), file["spectrum_shape_accidentals_AD$(AD)"][1:end])
    foreach(x -> push!(shape_alpha_neutron, x.N), file["spectrum_shape_alpha_neutron_AD$(AD)"][1:end])
    foreach(x -> push!(shape_amc, x.N), file["spectrum_shape_amc_AD$(AD)"][1:end])
    foreach(x -> push!(shape_fast_neutrons, x.N), file["spectrum_shape_fast_neutrons_AD$(AD)"][1:end])
    foreach(x -> push!(shape_lithium_helium, x.N), file["spectrum_shape_lithium_helium_AD$(AD)"][1:end])
    close(file)

    bg_shapes = (; shape_accidental, shape_alpha_neutron, shape_amc, shape_fast_neutrons, shape_lithium_helium)

    # background rates + uncertainties
    bg_rate_path = joinpath(data_basepath, "dayabay_background_rates.hdf5")
    data = h5open(bg_rate_path)
    rate_accidental = data["$(period)AD"][1][Symbol("AD$(AD)")]
    uncertainty_accidental = data["$(period)AD"][2][Symbol("AD$(AD)")]
    rate_lithium_helium = data["$(period)AD"][2][Symbol("AD$(AD)")]
    uncertainty_lithium_helium = data["$(period)AD"][4][Symbol("AD$(AD)")]
    rate_fast_neutrons = data["$(period)AD"][5][Symbol("AD$(AD)")]
    uncertainty_fast_neutrons = data["$(period)AD"][6][Symbol("AD$(AD)")]
    rate_amc = data["$(period)AD"][7][Symbol("AD$(AD)")]
    uncertainty_amc = data["$(period)AD"][8][Symbol("AD$(AD)")]
    rate_alpha_neutron = data["$(period)AD"][9][Symbol("AD$(AD)")]
    uncertainty_alpha_neutron = data["$(period)AD"][10][Symbol("AD$(AD)")]
    close(data)

    # daily detector data: lifetime + daily rate for accidentals
    daily_path = joinpath(data_basepath, "dayabay_daily_detector_data.hdf5")
    data = h5open(daily_path)
    eff_livetime = []
    acc_rate = []
    livetime = []
    foreach(x -> x[:n_det] == period ? push!(eff_livetime, x[:eff_livetime]) : 0, data["AD$(AD)"][1:end])
    foreach(x -> x[:n_det] == period ? push!(acc_rate, x[:rate_accidentals]) : 0, data["AD$(AD)"][1:end])
    foreach(x -> x[:n_det] == period ? push!(livetime, x[:livetime]) : 0, data["AD$(AD)"][1:end])
    #foreach(x -> x[:n_det] == 6 ? push!(lt, x[:livetime]) : 0, data["AD11"][1:end])

    close(data)

    eff_livetime = sum(livetime) / 60 / 60 / 24   # convert from seconds to days


    #counts_accidental = sum(eff_livetime .* acc_rate)
    #counts_lithium_helium = sum(eff_livetime .* rate_lithium_helium)
    #counts_fast_neutrons = sum(eff_livetime .* rate_fast_neutrons)
    #counts_amc = sum(eff_livetime .* rate_amc)
    #rate_alpha_neutron = sum(eff_livetime .* rate_alpha_neutron)

    lihe = (rate=rate_lithium_helium, shape=shape_lithium_helium, uncertainty=uncertainty_lithium_helium)
    amc = (rate=rate_amc, shape=shape_amc, uncertainty=uncertainty_amc)
    fast_neutrons = (rate=rate_fast_neutrons, shape=shape_fast_neutrons, uncertainty=uncertainty_fast_neutrons)
    alpha_neutron = (rate=rate_alpha_neutron, shape=shape_alpha_neutron, uncertainty=uncertainty_alpha_neutron)
    accidentals = (rate=rate_accidental, shape=shape_accidental, uncertainty=uncertainty_accidental)

    bg_dict = Dict()
    bg_dict["accidentals"] = accidentals
    bg_dict["lithium_helium"] = lihe
    bg_dict["alpha_neutron"] = alpha_neutron
    bg_dict["fast_neutrons"] = fast_neutrons
    bg_dict["amc"] = amc

    (;counts_ibd, E_bins_MeV, E_center_MeV, bg_dict, eff_livetime)
end

function get_iav_matrix()
    file = h5open("dayabay_data/detector_iav_matrix.hdf5")

    iav = file["iav_matrix"][]    # sums in dim=2 to 1
    close(file)
    iav
end

function get_lnsl_correction()
    # gives ratio of visible/true energy, hence multiply true energy to go to visible
    file = h5open("dayabay_data/detector_lsnl_curves.hdf5")

    f_nom = []
    pull0 = []
    pull1 = []
    pull2 = []
    pull3 = []
    E = []
    E0 = []
    E1 = []
    E2 = []
    E3 = []
    foreach(x -> (push!(E, x[1]), push!(f_nom, x[2])), file["nominal"][1:end])
    foreach(x -> (push!(E0, x[1]), push!(pull0, x[2])), file["pull0"][1:end])
    foreach(x -> (push!(E1, x[1]), push!(pull1, x[2])), file["pull1"][1:end])
    foreach(x -> (push!(E2, x[1]), push!(pull2, x[2])), file["pull2"][1:end])
    foreach(x -> (push!(E3, x[1]), push!(pull3, x[2])), file["pull3"][1:end])
    close(file)

    # subtract norm from pull term, can use multiplicative term for pull spectra; factor of zero means nominal
    rel_0 = pull0 .- f_nom
    rel_1 = pull1 .- f_nom
    rel_2 = pull2 .- f_nom
    rel_3 = pull3 .- f_nom

    @assert all(isapprox.(E, E0))
    @assert all(isapprox.(E0, E1))
    @assert all(isapprox.(E1, E2))
    @assert all(isapprox.(E2, E3))

    return (;E, rel_0, rel_1, rel_2, rel_3)

end

function get_priors()
    ## accidentals
    dict = YAML.load_file("dayabay_data/parameters/background_rate_scale_accidentals.yaml")
    # multiply for all ADs
    acc_scale_nom = dict["parameters"]["accidentals"][1]  # blow up to vector over all ADs
    acc_scale_unc = acc_scale_nom * 0.01   # percent error  # blow up to diagonal matrix

    len = length(detectors_6AD) + length(detectors_8AD) + length(detectors_7AD)
    acc_scale = Distributions.MvNormal(acc_scale_nom .*ones(len), Diagonal(acc_scale_unc^2 .* ones(len)))


    ## amc
    dict = YAML.load_file("dayabay_data/parameters/background_rate_uncertainty_scale_amc.yaml")

    amc_scale_nom = dict["parameters"]["amc"][1]
    amc_scale_unc = dict["parameters"]["amc"][2]  # absolute

    amc_unc_scale = Distributions.Normal(amc_scale_nom, amc_scale_unc)


    ## site correlated
    dict = YAML.load_file("dayabay_data/parameters/background_rate_uncertainty_scale_site.yaml")

    lihe_nom = dict["parameters"]["lithium_helium"][1]
    lihe_unc = dict["parameters"]["lithium_helium"][2]
    fast_n_nom = dict["parameters"]["fast_neutrons"][1]
    fast_n_unc = dict["parameters"]["fast_neutrons"][2]

    len = length(period_list) * length(EH_list)
    fast_n_unc_scale = Distributions.MvNormal(fast_n_nom .* ones(len), Diagonal(fast_n_unc^2 .* ones(len)))
    lihe_unc_scale = Distributions.MvNormal(lihe_nom .* ones(len), Diagonal(lihe_unc .*ones(len)))


    ## uncorrelated
    # dicttype keeps the order in which the yaml is written
    dict = YAML.load_file("dayabay_data/parameters/background_rates_uncorrelated.yaml", dicttype=OrderedDict{String,Any})

    alpha_n_nom = Float64[]
    alpha_n_unc = Float64[]

    for k in period_list
        an = dict["parameters"]["alpha_neutron"][k]
        for (k, v) in an
            push!(alpha_n_nom, v[1])
            push!(alpha_n_unc, v[2])
        end
    end

    alpha_n_rate = Distributions.MvNormal(alpha_n_nom, Diagonal(alpha_n_unc.^2))


    ## AD efficiency factor / energy scale: per AD, constant over periods
    # is correlated within AD
    dict = YAML.load_file("dayabay_data/parameters/detector_relative.yaml")

    eff_nom = Float64(dict["parameters"]["detector_relative"]["energy_scale_factor"][1])
    eff_unc = Float64(dict["parameters"]["detector_relative"]["energy_scale_factor"][2] * 0.01)  # percent
    e_scale_nom =  dict["parameters"]["detector_relative"]["energy_scale_factor"][1]
    e_scale_unc = dict["parameters"]["detector_relative"]["energy_scale_factor"][2] * 0.01  # percent

    scale = hcat([[eff_unc^2, eff_unc * e_scale_unc], [eff_unc * e_scale_unc, e_scale_unc^2]]...)

    corr_mat = hcat(dict["correlations"]["detector_relative"]["matrix"]...)
    # correlation(i, j) = covariance(i, j) / sqrt(var_i * var_j)) -> invert to get covariance matrix for MvNormal
    cov_mat = corr_mat .* scale
    println(scale)
    println(cov_mat)
    eff_eres_nom = Vector([eff_nom, e_scale_nom])
    eff_eres_AD11 = Distributions.MvNormal(eff_eres_nom, cov_mat)
    eff_eres_AD12 = Distributions.MvNormal(eff_eres_nom, cov_mat)
    eff_eres_AD21 = Distributions.MvNormal(eff_eres_nom, cov_mat)
    eff_eres_AD22 = Distributions.MvNormal(eff_eres_nom, cov_mat)
    eff_eres_AD31 = Distributions.MvNormal(eff_eres_nom, cov_mat)
    eff_eres_AD32 = Distributions.MvNormal(eff_eres_nom, cov_mat)
    eff_eres_AD33 = Distributions.MvNormal(eff_eres_nom, cov_mat)
    eff_eres_AD34 = Distributions.MvNormal(eff_eres_nom, cov_mat)


    ## energy resolution, shared across all ADs
    dict = YAML.load_file("dayabay_data/parameters/detector_eres.yaml")
    a_nom = dict["parameters"]["eres"]["a_nonuniform"][1]
    a_unc = a_nom * dict["parameters"]["eres"]["a_nonuniform"][2] * 0.01   # percent

    b_nom = dict["parameters"]["eres"]["b_stat"][1]
    b_unc = b_nom * dict["parameters"]["eres"]["b_stat"][2] * 0.01   # percent

    c_nom = dict["parameters"]["eres"]["c_noise"][1]
    c_unc = c_nom * dict["parameters"]["eres"]["c_noise"][2] * 0.01   # percent

    eres_a = Distributions.Normal(a_nom, a_unc)
    eres_b = Distributions.Normal(b_nom, b_unc)
    eres_c = Distributions.Normal(c_nom, c_unc)


    ## iav off diagonal scaling, one for each AD, constant over periods
    dict = YAML.load_file("dayabay_data/parameters/detector_iav_offdiag_scale.yaml")
    iav_scale_nom = dict["parameters"]["iav_offdiag_scale_factor"][1]
    iav_scale_unc = dict["parameters"]["iav_offdiag_scale_factor"][2]
    len = length(full_setup)
    iav_offdiag_scale = Distributions.MvNormal(iav_scale_nom .* ones(len), iav_scale_unc^2 .* Diagonal(ones(len)))

    ## lsnl correction, pull parameters
    dict = YAML.load_file("dayabay_data/parameters/detector_lsnl.yaml")
    lsnl_scale_nom = dict["parameters"]["lsnl_scale_a"][1]
    lsnl_scale_unc = dict["parameters"]["lsnl_scale_a"][2]
    lsnl_pull = Distributions.MvNormal(lsnl_scale_nom .* ones(4), lsnl_scale_unc^2 .* ones(4))


    priors = (
        amc_unc_scale=amc_unc_scale,  # done
        acc_scale=acc_scale,   # done
        lihe_unc_scale=lihe_unc_scale,  # done
        fast_n_unc_scale=fast_n_unc_scale,   # done
        alpha_n_rate=alpha_n_rate,   # done
        eres_a=eres_a,  # done
        eres_b=eres_b,  # done
        eres_c=eres_c,  # done
        eff_eres_AD11=eff_eres_AD11,
        eff_eres_AD12=eff_eres_AD12,
        eff_eres_AD21=eff_eres_AD21,
        eff_eres_AD22=eff_eres_AD22,
        eff_eres_AD31=eff_eres_AD31,
        eff_eres_AD32=eff_eres_AD32,
        eff_eres_AD33=eff_eres_AD33,
        eff_eres_AD34=eff_eres_AD34,
        iav_offdiag_scale=iav_offdiag_scale   # done
    )
end

function get_params()
    ## accidentals
    dict = YAML.load_file("dayabay_data/parameters/background_rate_scale_accidentals.yaml")
    # multiply for all ADs
    acc_scale_nom = dict["parameters"]["accidentals"][1]  # blow up to vector over all ADs

    len = length(detectors_6AD) + length(detectors_8AD) + length(detectors_7AD)
    acc_scale = acc_scale_nom .*ones(len)


    ## amc
    dict = YAML.load_file("dayabay_data/parameters/background_rate_uncertainty_scale_amc.yaml")

    amc_scale_nom = dict["parameters"]["amc"][1]

    amc_unc_scale = amc_scale_nom
    #TODO : possibly solve this by HierarchicalDistributions
    #hd = HierarchicalDistribution(
    #    v -> BAT.NamedTupleDist(rate=Distributions.Normal(0.277, 0.11 * v.scale)),
    #    BAT.NamedTupleDist(scale=Normal(0.0, 1.0))
    #)
    #amc_rate = 


    ## site correlated
    dict = YAML.load_file("dayabay_data/parameters/background_rate_uncertainty_scale_site.yaml")

    lihe_nom = dict["parameters"]["lithium_helium"][1]
    fast_n_nom = dict["parameters"]["fast_neutrons"][1]

    len = length(period_list) * length(EH_list)
    fast_n_unc_scale = fast_n_nom .* ones(len)
    lihe_unc_scale = lihe_nom .* ones(len)


    ## uncorrelated
    # dicttype keeps the order in which the yaml is written
    dict = YAML.load_file("dayabay_data/parameters/background_rates_uncorrelated.yaml", dicttype=OrderedDict{String,Any})

    alpha_n_nom = Float64[]

    for k in period_list
        an = dict["parameters"]["alpha_neutron"][k]
        for (k, v) in an
            push!(alpha_n_nom, v[1])
        end
    end

    alpha_n_rate = alpha_n_nom


    ## AD efficiency factor / energy scale: per AD, constant over periods
    # is correlated within AD
    dict = YAML.load_file("dayabay_data/parameters/detector_relative.yaml")

    eff_nom = Float64(dict["parameters"]["detector_relative"]["energy_scale_factor"][1])
    e_scale_nom =  dict["parameters"]["detector_relative"]["energy_scale_factor"][1]

    eff_eres_nom = Vector([eff_nom, e_scale_nom])
    eff_eres_AD11 = eff_eres_nom
    eff_eres_AD12 = eff_eres_nom
    eff_eres_AD21 = eff_eres_nom
    eff_eres_AD22 = eff_eres_nom
    eff_eres_AD31 = eff_eres_nom
    eff_eres_AD32 = eff_eres_nom
    eff_eres_AD33 = eff_eres_nom
    eff_eres_AD34 = eff_eres_nom


    ## energy resolution, shared across all ADs
    dict = YAML.load_file("dayabay_data/parameters/detector_eres.yaml")
    a_nom = dict["parameters"]["eres"]["a_nonuniform"][1]
    b_nom = dict["parameters"]["eres"]["b_stat"][1]
    c_nom = dict["parameters"]["eres"]["c_noise"][1]

    eres_a = a_nom
    eres_b = b_nom
    eres_c = c_nom


    ## iav off diagonal scaling, one for each AD, constant over periods
    dict = YAML.load_file("dayabay_data/parameters/detector_iav_offdiag_scale.yaml")
    iav_scale_nom = dict["parameters"]["iav_offdiag_scale_factor"][1]
    len = length(full_setup)
    iav_offdiag_scale = iav_scale_nom .* ones(len)

    ## lsnl correction, pull parameters
    dict = YAML.load_file("dayabay_data/parameters/detector_lsnl.yaml")
    lsnl_scale_nom = dict["parameters"]["lsnl_scale_a"][1]
    lsnl_pull = lsnl_scale_nom .* ones(4)
    params = (;
        amc_unc_scale,
        acc_scale,
        lihe_unc_scale,
        fast_n_unc_scale,
        alpha_n_rate,
        eres_a,
        eres_b,
        eres_c,
        eff_eres_AD11,
        eff_eres_AD12,
        eff_eres_AD21,
        eff_eres_AD22,
        eff_eres_AD31,
        eff_eres_AD32,
        eff_eres_AD33,
        eff_eres_AD34,
        iav_offdiag_scale
    )
    return params
end

function get_assets()
end

    

function extract_energy_resolution()
end

```

```julia
#effective livetime = livetime * efficiency, in days. multiply with rates [days-1] to get counts
params = get_params()

period = "6AD"
AD = "AD11"
EH = AD_to_EH(AD)
idx = findall(x-> x=="$(period)$(AD)", detector_list)[1]
output = extract_for_AD_period(11, 6)
bg_dict = output.bg_dict
eff_livetime = output.eff_livetime
accidentals = bg_dict["accidentals"]
acc_rate_unc = accidentals.uncertainty * params.acc_scale

amc = bg_dict["amc"]
lihe = bg_dict["lithium_helium"]
fast_n = bg_dict["fast_neutrons"]
alpha_n = bg_dict["alpha_neutron"]

acc_counts = eff_livetime * accidentals.rate * params.acc_scale[idx] .* accidentals.shape 

amc_counts = eff_livetime * amc.rate * (1 + amc.uncertainty * params.amc_unc_scale) .* amc.shape

period_EH_idx = findall(x-> x == "$(period)_$(idx)", period_EH_list)[1]
lihe_counts = eff_livetime * lihe.rate * (1 + lihe.uncertainty * params.lihe_unc_scale[period_EH_idx]) .* lihe.shape

fast_n_counts = eff_livetime * fast_n.rate * (1 + fast_n.uncertainty * params.fast_n_unc_scale[period_EH_idx]) .* fast_n.shape

alpha_n_counts = eff_livetime * alpha_n.rate .* alpha_n.shape

```

```julia
sum(alpha_n_counts)
```

```julia
eff_livetime = output.eff_livetime
rates = Dict()
for (k, v) in output.bg_dict
    rates[k] = v.rate
end
rates

```

```julia
eff_livetime / 60 / 60 / 24
```

```julia
data_basepath = "dayabay_data/dayabay_dataset"
period = 6
daily_path = joinpath(data_basepath, "dayabay_daily_detector_data.hdf5")
eff_livetime_per_AD = []
data = h5open(daily_path)
for AD in detectors_6AD
    eff_livetime = []
    foreach(x -> x[:n_det] == period ? push!(eff_livetime, x[:eff_livetime]) : 0, data["AD$(AD[3:end])"][1:end])
    push!(eff_livetime_per_AD, sum(eff_livetime))
end


```

```julia
eff_livetime_per_AD
```

```julia
datasets = HDF5.get_datasets(data)

datasets[1][1][:n_det]
```

```julia
dict = YAML.load_file("dayabay_data/parameters/background_rates_uncorrelated.yaml", dicttype=OrderedDict{String,Any})

for k in keys
    println(dict["parameters"]["alpha_neutron"][k])
    break
end

keys = ["6AD", "8AD", "7AD"]

detectors_6AD = []
detectors_8AD = []
detectors_7AD = []

alpha_n_nom = []
alpha_n_unc = []

for k in keys
    an = dict["parameters"]["alpha_neutron"][k]
    for (k, v) in an
        push!(alpha_n_nom, v[1])
        push!(alpha_n_unc, v[2])
    end

    
end
for (k, v) in dict["parameters"]["alpha_neutron"]["6AD"]
    push!(detectors_6AD, k)
end
for (k, v) in dict["parameters"]["alpha_neutron"]["8AD"]
    push!(detectors_8AD, k)
end
for (k, v) in dict["parameters"]["alpha_neutron"]["7AD"]
    push!(detectors_7AD, k)
end

```

```julia
alpha_n_nom
```

```julia
output = extract_for_AD_period(11, 8)
```

```julia

```

```julia

```

```julia

```

```julia
output.bg_dict["accidentals"].counts
```

```julia
dict["parameters"]["accidentals"]
```

<!-- #region -->
### Correlation overview


#### backgrounds
accidental: uncorrelated between all ADs

AmC: fully correlated, one parameter for all ADs

Lithium-Helium: correlated within site, i.e. three parameters (EH1, EH2, EH3)

fast neutrons: same as Lithium-Helium



#### detector effects

detector_relative: one for each AD

extra/detector_absolute: ignore for now, global normalisation
<!-- #endregion -->

```julia
# Reactors / Baselines, etc
exp_dict = Dict("Detector" => ["AD1", "AD2", "AD3", "AD8", "AD4", "AD5", "AD6", "AD7"],
            "EH" => ["EH1", "EH1", "EH2", "EH2", "EH3", "EH3", "EH3", "EH3"],
            "6" => [true, true, true, false, true, true, true, false],
            "8" => [true, true, true, true, true, true, true, true],
            "7" => [true, false, true, true, true, true, true, true],
)
df_exp = DataFrame(exp_dict);
```

```julia
df_exp[!, "6"]
```

```julia
output = extract_for_AD_period(11, 6)

```

```julia
file = h5open("dayabay_data/detector_lsnl_curves.hdf5")
file

#E = []
f_nom = []
pull0 = []
pull1 = []
pull2 = []
pull3 = []
Enom = []
E0 = []
E1 = []
E2 = []
E3 = []
foreach(x -> (push!(Enom, x[1]), push!(f_nom, x[2])), file["nominal"][1:end])
foreach(x -> (push!(E0, x[1]), push!(pull0, x[2])), file["pull0"][1:end])
foreach(x -> (push!(E1, x[1]), push!(pull1, x[2])), file["pull1"][1:end])
foreach(x -> (push!(E2, x[1]), push!(pull2, x[2])), file["pull2"][1:end])
foreach(x -> (push!(E3, x[1]), push!(pull3, x[2])), file["pull3"][1:end])

@assert all(isapprox.(E0, E1))
@assert all(isapprox.(E1, E2))
@assert all(isapprox.(E2, E3))
```

```julia
f_nom
```

```julia
data = h5open("dayabay_data/dayabay_dataset/dayabay_background_spectra_6AD.hdf5")
data["spectrum_shape_accidentals_AD11"][]
close(data)
```

```julia
f = Figure()
ax = Axis(f[1, 1])

scatter!(output.E_center_MeV, lihe_hist)
f
```

```julia
data_basepath = "dayabay_data/parameters"

final_binning = readdlm(joinpath(data_basepath, "final_erec_bin_edges.tsv"))[2:end]
E_center_MeV = output.E_center_MeV

rebin_idx = searchsortedlast.(Ref(final_binning), E_center_MeV)
```

```julia
searchsortedlast.(Ref(energy_bins), E_prompt_reco_binc)
```

```julia
data = h5open("dayabay_data/neutrino_rate.hdf5")
data["R1"][]
```

```julia

```

```julia

```

```julia

```

```julia

```

```julia
Newtrinos.reactor_flux.configure().nominal_flux(5.4)
```

```julia
db = Newtrinos.dayabay_rewrite.configure()
E_prompt_binc = db.assets.E_prompt_binc
E_prompt_edges = db.assets.E_prompt;
```

```julia
EH = 1
period = [1]
db.assets.L_arrs[EH][period]
```

```julia
db.assets.flux_weights_bar_osc[EH]
```

```julia
prompt_binc = (E_prompt_edges[1:end-1] + E_prompt_edges[2:end]) / 2;
```

```julia
# map neutrino energy to positron energy: E_positron = E_nu - 0.782 in MeV
# apply non-linearity function of the scintillator material: E_prompt = E_positron * non_linearity(E_positron)
# smear by energy resolution
# apply energy scale uncertainty


# energy resolution: get relative uncertainty from quadratic form, combining eres_a,b,c
#       use E_prompt vector to create adhoc matrix
```

```julia
experiments = (dayabay = Newtrinos.dayabay_rewrite.configure(),)

vars_to_scan = OrderedDict()
vars_to_scan[:θ₁₃] = 31
vars_to_scan[:Δm²₃₁] = 31

likelihood = Newtrinos.generate_likelihood(experiments);

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
physics = experiments.dayabay.physics
assets = experiments.dayabay.assets

@reset priors.Δm²₃₁ = Uniform(0.0023, 0.0028)
@reset priors.θ₁₃ = Uniform(0.13, 0.16)

@reset priors.θ₁₂ = p.θ₁₂
@reset priors.θ₂₃ = p.θ₂₃
@reset priors.θ₁₃ = p.θ₁₃
@reset priors.δCP = p.δCP
@reset priors.Δm²₂₁ = p.Δm²₂₁
@reset priors.Δm²₃₁ = p.Δm²₃₁
#@reset priors.pulls = zeros(25)
@reset priors.background_norm = 0.
@reset priors.per_EH_norm = ones(3)

Newtrinos.dayabay_rewrite.get_expected(p, physics, assets)


# define energies such that we can use the IAV correction as-is
E_prompt_edges = Vector(LinRange(0, 12, 241))
E_prompt_binc = (E_prompt_edges[1:end-1] + E_prompt_edges[2:end]) / 2

E_antinu_binc = E_prompt_binc .+ 0.782
E_antinu_edges = E_prompt_edges .+ 0.782

flux_xsec = @. physics.flux.nominal_flux(E_antinu_binc) * physics.xsec.xsec(E_antinu_binc)




# flux_xsec = @. assets.nom_flux * assets.xsec_eval
nonlinearity = assets.nonlinearity;

```

```julia
assets.E_lsnl
```

```julia
f = h5open("/home/iwsatlas1/kuhlmann/DEMOS/neutrinos/osc/dayabay_data/detector_iav_matrix.hdf5")
```

```julia
iav = transpose(read(f, "iav_matrix"))
```

```julia
close(f)
```

```julia
heatmap(log10.(iav))
```

```julia
delta_n = zeros(240)

delta_n[100] = 1

sum(iav * delta_n)
```

```julia
f = Figure()
ax = Axis(f[1, 1])

lines!(iav * delta_n)
lines!(transpose(iav) * delta_n)

xlims!(ax, 90, 110)

f
```

```julia
#binc = LinRange(0.025, 11.975, 240)[:]
#iav_interp = linear_interpolation((binc, binc), log10.(iav), extrapolation_bc=-Inf)
```

```julia
function energy_resolution(E_arr_smear_local, smear_arr_in, sigma_arr; width=100, E_scale=1.0, E_bias=0.0)
    l = length(smear_arr_in)
    T_acc = promote_type(eltype(E_arr_smear_local), eltype(sigma_arr), eltype(smear_arr_in), typeof(E_scale))
    out = zeros(T_acc, l)

    for i in 1:l

        e_center = E_arr_smear_local[i] * E_scale + E_bias

        norm_val = zero(T_acc)
        sum_val = zero(T_acc)

        j_min_loop = max(1, i - width)
        j_max_loop = min(l, i + width)

        for j in j_min_loop:j_max_loop

            coeff = (1 / sigma_arr[j]) * exp(-0.5 * ((e_center - E_arr_smear_local[j]) / sigma_arr[j])^2)
            norm_val += coeff
            sum_val += coeff * smear_arr_in[j]

        end

        if norm_val > 1e-10
            out[i] = sum_val / norm_val
        end

    end
    return out
end
```

```julia
energy_bins = assets.energy_bins

ebinc = (energy_bins[1:end-1] + energy_bins[2:end]) / 2;

observed = assets.observed
```

```julia
delta_n = zeros(240)

delta_n[230] = 1.

sum(iav * delta_n)
```

```julia
iav * flux_xsec
```

```julia
m_p = 938.272088
m_n = 939.565421
m_e =   0.510998
one_over_m_p = 1 / m_p

delta = (m_n^2 - m_p^2  - m_e^ 2 ) / 2 / m_p
function s(E_nu)
    2 * m_p * E_nu + m_p^2
end
function E_nu_CM(s)
    (s - m_p^2) / 2 / sqrt(s)
end
function E_e_CM(s)
    (s - m_n^2 + m_e^2) / 2 / sqrt(s)
end

function p_e_CM(s)
    sqrt((s - (m_n - m_e)^2) * (s - (m_n + m_e)^2)) / 2 / sqrt(s)
end

function E_inf_sup(E_nu)
    _s = s(E_nu)
    _p_e_CM = p_e_CM(_s)
    _E_nu_CM = E_nu_CM(_s)
    _E_e_CM = E_e_CM(_s)
    E_nu_minus_delta = E_nu - delta
    prefac =  one_over_m_p * _E_nu_CM
    (E_inf = E_nu_minus_delta - prefac * (_E_e_CM + _p_e_CM),
    E_sup = E_nu_minus_delta - prefac * (_E_e_CM - _p_e_CM))
end

bounds = E_inf_sup(0.7)

diff = bounds.E_sup - bounds.E_inf
```

```julia
#do computation with provided IRF matrix containing all necessary things

eres = assets.eres
nu_energy = assets.E_antinu_binc
flux_xsec = @. physics.flux.nominal_flux(nu_energy) * physics.xsec.xsec(nu_energy)


response = eres * flux_xsec
E_response = assets.E_prompt_binc
binned_response = zeros(length(energy_bins) - 1)

bin_idxs =  searchsortedlast.(Ref(energy_bins), E_response)

for i in eachindex(binned_response)
    binned_response[i] = sum(response[bin_idxs .== i])
end
```

```julia
binned_response
```

```julia
(- m_n + m_p - m_e) / 2
```

```julia
f = Figure()
ax = Axis(f[1, 1], xlabel="true prompt energy", ylabel="reco prompt energy")
x = LinRange(0, 5, 1000)

lines!(x, x .* nonlinearity.(x))

lines!(x, x, label="1:1")

f
```

```julia
E_prompt_binc
```

```julia
IBD_counts = CSV.read("/home/iwsatlas1/kuhlmann/DEMOS/neutrinos/osc/dayabay_data/EH1_ibd_counts.txt", DataFrame)
```

```julia
ibd_counts = IBD_counts[!, "# counts"]
```

```julia
sum(assets.Npred_EH_nooscs[1])

```

```julia
assets.dfIBD_dict["dfIBD_EH1"][!, "Nobs_6AD"] .+ assets.dfIBD_dict["dfIBD_EH1"][!, "Nobs_8AD"] .+ assets.dfIBD_dict["dfIBD_EH1"][!, "Nobs_7AD"]
```

```julia
pred_best_fit = assets.dfIBD_dict["dfIBD_EH1"][!, "Npred_6AD"] .+ assets.dfIBD_dict["dfIBD_EH1"][!, "Npred_8AD"] .+ assets.dfIBD_dict["dfIBD_EH1"][!, "Npred_7AD"]

```

```julia
assets.dfIBD_dict["dfIBD_EH1"]
```

```julia
f = Figure()

ax = Axis(f[1, 1])

scatter!(ebinc, assets.total_observed[1], label="IBD w/ osc + bg")

scatter!(ebinc, assets.observed, label="IBD pred w/o osc")

scatter!(ebinc, sum(assets.Npred_EH_oscs[1]), label="IBD pred w/ osc")

scatter!(ebinc, sum(assets.Npred_EH_nooscs[1]), markersize=10, label="IBD pred w/o osc")


scatter!(ebinc, pred_best_fit, markersize=6)



axislegend(ax)



f
```

```julia
f = Figure()
ax = Axis(f[1, 1], xlabel="energy [MeV]", ylabel="Counts, arbitrary scale", title="EH1, predicted no oscillations")

# E_antinu_binc = assets.E_antinu_binc

integrand(E, p) = physics.flux.nominal_flux(E) * physics.xsec.xsec(E)



#flux_xsec = zeros(240)
flux_xsec = @. physics.flux.nominal_flux(E_antinu_binc) * physics.xsec.xsec(E_antinu_binc)
#for i in eachindex(flux_xsec)
#    prob = IntegralProblem(integrand, (E_antinu_edges[i], E_antinu_edges[i+1]))
#    flux_xsec[i] = solve(prob, QuadGKJL()).u
#end


# E_prompt_binc = E_antinu_binc .- 0.7

#lines!(E_antinu_binc, flux_xsec, linewidth=1, label="flux x cross-section, neutrino energy")
# iav_flux = zeros(flux_xsec.size)

iav_flux = iav * flux_xsec;


lsnl_idx = E_prompt_binc .>= assets.E_lsnl[1]

E_lsnl = nonlinearity.(E_prompt_binc) .* E_prompt_binc


#lines!(E_prompt_binc, eres_flux)

#lines!(E_lsnl, flux_xsec, linewidth=1, label="flux x cross section, E_lsnl")



eres_a = 0.016
eres_b = 0.081
eres_c = 0.026
eres_lsnl = @. sqrt(E_lsnl^2 * eres_a^2 + E_lsnl * eres_b^2 + eres_c^2)


#lines!(E_lsnl, iav_flux, label="flux after iav")

iav_response = energy_resolution(E_lsnl, iav_flux, eres_lsnl, width=100)
response = energy_resolution(E_lsnl, flux_xsec, eres_lsnl, width=100)

iav_response = iav * iav_response

#lines!(E_lsnl, iav_response, label="flux iav + eres")


bin_idxs =  searchsortedlast.(Ref(energy_bins), E_lsnl)
result = zeros(length(energy_bins) - 1)
iav_result = zeros(length(energy_bins) - 1)

idxs = searchsortedlast.(Ref(energy_bins), E_prompt_binc)
proper_results = zeros(length(energy_bins) - 1)

for i in eachindex(result)
    result[i] += sum(response[bin_idxs .== i]) / sum(bin_idxs .== i) #(energy_bins[i+1] - energy_bins[i])
    iav_result[i] += sum(iav_response[bin_idxs .== i]) / sum(bin_idxs .== i) #(energy_bins[i+1] - energy_bins[i])
    proper_results[i] = sum(ibd_counts[idxs .== i]) / sum(idxs .== i)

end

#scatter!(ebinc, result, label="flux * xsec, binned")
scatter!(ebinc, iav_result, label="iav+nonlinearity+eres")
scatter!(ebinc, assets.observed / sum(assets.observed) * sum(result), label="total observed events")
#scatter!(ebinc, observed/ sum(observed) * sum(result), label="observed scaled")# * sum(iav_result))
#scatter!(ebinc, binned_response / sum(binned_response) * sum(result), label="IRF matrix")

#scatter!(E_prompt_binc, ibd_counts ./maximum(ibd_counts) .* maximum(result))
scatter!(ebinc, proper_results ./ sum(proper_results) * sum(result), label="zenodo")
#scatter!(ebinc, assets.total_observed[1] / sum(assets.total_observed[1]) * sum(result), )

#vlines!(energy_bins, alpha=0.3, color=:black)
#xlims!(ax, 7, 12)
axislegend(ax, position=:rt)
f
#save("db_irf_comparison.png", f, dpi=300)
```

```julia
#increase resolution of iav along axis 1, s.t. we can use a finer grid on which to evaluate flux x cross-section
sum(iav, dims=1)
```

```julia
integrand(E, p) = 10^iav_diff_interp(E, E_prompt_binc[120])
prob = IntegralProblem(integrand, (0, E_prompt_binc[120]))
```

```julia

```

```julia
f = Figure()
ax = Axis(f[1, 1])
lines!(iav_diff[:, 120])
xlims!(ax, 100, 121)

f
```

```julia
iav_diff[120, 120]
```

```julia
integrand(E_prompt_edges[119], 2)
```

```julia
sol = solve(prob, QuadGKJL())
```



```julia
norms = []

for i in eachindex(E_prompt_binc)
    integrand(E_in, E_out) = 10^iav_diff_interp(E_in, E_prompt_binc[i])
    prob = IntegralProblem(integrand, (0, E_prompt_binc[i]))
    sol = solve(prob, QuadGKJL())
    push!(norms, sol.u)
end
```

```julia
UpperTriangular(iav)
```

```julia

```

```julia
size(iav)
```

```julia
iav_diff = zeros(240, 240)

for i in 1:240
    iav_diff[i, :] = iav[i, :] ./ (E_prompt_edges[2:end] .- E_prompt_edges[1:end-1])
end

interp = Interpolator(E_prompt_binc, iav_diff[:, 50])
```

```julia
interp_cdf = Interpolator(E_prompt_edges[2:end], cumsum(iav, dims=1)[:, 120])
```

```julia
interp_pdf(x) = ForwardDiff.derivative(interp_cdf, x)
```

```julia
interp_pdf(2)
```

```julia
f = Figure()
ax = Axis(f[1, 1], title="$(E_prompt_binc[50])")

lines!(E_prompt_binc, iav_diff[:, 50])
x = LinRange(0, 10, 10000)
lines!(x, interp_pdf.(x))
lines!(x, interp.(x))
#xlims!(5.5, 6.1)
f
```

```julia

```

```julia

```

```julia

```

```julia

```

```julia

```

```julia
E_prompt_binc[21]
```

```julia
assets.E_lsnl
```

```julia
sum(iav, dims=1)[:][20]
```

```julia
heatmap(E_prompt_binc, E_prompt_binc, log10.(iav_diff))
```

```julia
iav_diff_interp = linear_interpolation((E_prompt_binc, E_prompt_binc), log10.(iav_diff), extrapolation_bc=-Inf)
```

```julia
energy_bins
```

```julia
for i in eachindex(result)
    println(sum(bin_idxs .==i))
end
```

```julia
assets = experiments.dayabay.assets
physics = experiments.dayabay.physics
expected = Newtrinos.dayabay_rewrite.get_expected(p, physics, assets)
```

```julia
predicted_no_osc = experiments.dayabay.assets.predicted_nooscs
official_no_osc = experiments.dayabay.assets.Npred_EH_nooscs
energy_bins = experiments.dayabay.assets.energy_bins
e_binc = (energy_bins[2:end] + energy_bins[1:end-1]) / 2;
```

```julia
sum(sum(official_no_osc[:])[:])
```

```julia
assets.observed
```

```julia
assets.expected
```

```julia
sum(assets.Npred_EH_nooscs[1])
```

```julia
scatter(assets.nom_flux .* assets.xsec_eval)
```

```julia
markersize = 10

norm = experiments.dayabay.assets.norm
f = Figure()
ax = Axis(f[1, 1], xlabel="reconstructed prompt energy [MeV]",)
scatter!(e_binc, assets.observed, color="blue", markersize=markersize, marker=:utriangle)
scatter!(e_binc, expected[1:26] / sum(expected) * sum(assets.observed))
#scatter!(e_binc, assets.observed[27:52], color="red", markersize=markersize, marker=:utriangle)
#scatter!(e_binc, expected[27:52])
#scatter!(e_binc, assets.observed[53:end], color="green", markersize=markersize, marker=:utriangle)
#scatter!(e_binc, expected[53:end])
#scatter!(e_binc, sum(official_no_osc[3]), color="green", markersize=markersize, marker=:utriangle)
#scatter!(e_binc, sum(sum(official_no_osc[:])[:]), color="black", markersize=markersize, marker=:utriangle, label="DayaBay")
#scatter!(e_binc, sum(predicted_no_osc[1]) .*norm, color="blue", markersize=markersize, marker=:xcross)
##scatter!(e_binc, sum(predicted_no_osc[2]) .*norm, color="red", markersize=markersize, marker=:xcross)
#scatter!(e_binc, sum(predicted_no_osc[3]) .*norm, color="green", markersize=markersize, marker=:xcross)
#scatter!(e_binc, sum(sum(predicted_no_osc[:])[:]) .*norm, color="black", markersize=markersize, marker=:xcross, label="Newtrinos.jl")
#axislegend(ax)
f
```

```julia
scatter(assets.E_antinu_binc, assets.nom_flux)
```

```julia
markersize = 10
assets = experiments.dayabay.assets
norm = experiments.dayabay.assets.norm
expected = Newtrinos.dayabay_rewrite.get_expected(p, experiments.dayabay.physics, experiments.dayabay.assets)
l = 26
f = Figure()
ax = Axis(f[1, 1], xlabel="reconstructed prompt energy [MeV]",)
for i in 1:3
    scatter!(e_binc, assets.observed[(i-1)*l + 1: i*l])
    scatter!(e_binc, expected[(i-1)*l + 1: i*l])
end
#axislegend(ax)
f
```

```julia
ass = Newtrinos.dayabay_rewrite.configure().assets
```

```julia
@time Newtrinos.dayabay_rewrite.get_expected(p, experiments.dayabay.physics, experiments.dayabay.assets)
```

```julia
assets.observed
```

```julia
mle = Newtrinos.find_mle(likelihood, distprod(priors), p)
```

```julia
db.assets.predicted_nooscs[1][1]
```

```julia
best_fitting_exp = Newtrinos.dayabay_rewrite.get_expected(mle[3], experiments.dayabay.physics, experiments.dayabay.assets)
```

```julia
mle[3].background_norm
```

```julia
m = Distributions.mean(experiments.dayabay.forward_model(mle[3]))
v = Distributions.var(experiments.dayabay.forward_model(mle[3]))
```

```julia
assets = experiments.dayabay.assets
```

```julia
mle[3].norm
```

```julia
db.assets.norm
```

```julia
inch = 96
pt = 4/3
cm = inch / 2.54
f = Figure(size=(15cm, 25cm), dpi=300)
size_per_EH = length(assets.observed)


for i in 1:3
    min = Int(1 + (i-1) * size_per_EH)
    max = Int(i * size_per_EH)
    _v = v[min:max]
    _m = m[min:max]
    data = assets.observed[min:max]
    println(data)
    _exp = best_fitting_exp[min:max]
    ax = Axis(f[i*2-1, 1], title="EH $(i)")
    plot!(ax, assets.energy, data, color=:black, label="Observed")
    #plot!(ax, assets.energy, _exp, color=:red, label="naive expectation")
    stephist!(ax, assets.energy, weights=m[min:max], bins=assets.energy_bins, label="Expected")
    barplot!(ax, assets.energy, _m .+ sqrt.(_v), width=diff(assets.energy_bins), gap=0, fillto= _m.-sqrt.(_v), label="standard deviation", alpha=0.5)

    rowsize!(f.layout, 1 + (i-1) * 2, Relative(1/4))
    ax.xticksvisible = false
    ax.xticklabelsvisible = false

    ax2 = Axis(f[i*2,1])
    plot!(ax2, assets.energy, data ./ _m, color=:black, label="Observed")
    hlines!(ax2, 1, label="Expected")
    barplot!(ax2, assets.energy, 1 .+ sqrt.(_v) ./ _m, width=diff(assets.energy_bins), gap=0, fillto= 1 .- sqrt.(_v)./_m, alpha=0.5, label="Standard Deviation")
    ylims!(ax2, 0.95, 1.05)

    
    ax2.xlabel="Eₚ (MeV)"
    ax2.ylabel="Counts/Expected"

    xlims!(ax, minimum(assets.energy_bins), maximum(assets.energy_bins))
    xlims!(ax2, minimum(assets.energy_bins), maximum(assets.energy_bins))
    break
    #ylims!(ax, 0, 60000)
end
f
# save("db.png", f, dpi=300)
```

```julia
function deltachi2(exp, obs)
    2 * sum(@. obs * log(obs / exp) + exp - obs)
end


deltachi2(m, assets.observed) / (length(assets.observed) - 25 + 1 + 2)
```

```julia
assets.norm
```

```julia
length(size(zeros((25, 2))))
```

```julia
E = ass.energy;
```

```julia

```

```julia
sum(ass.Npred_EH_oscs)
```

```julia
bg_per_period = sum(stack([ass.bkg_templates["Seven_1"], ass.bkg_templates["Eight_1"], ass.bkg_templates["Seven_1"]]), dims=2)[:, 1, :]
```

```julia
ass.Npred_EH_oscs[1]
```

```julia
ass.observed[1]
```

```julia
ass.L_arrs
```

```julia
ass.L_arrs
```

```julia
f = Figure()
ax = Axis(f[1, 1])

scatter!(E, sum(ass.bkg_EH[1], dims=2)[:], label="background")
scatter!(E, sum(ass.Npred_EH_nooscs, dims=1)[1], label="predicted, no oscs")
scatter!(E, ass.observed[1], label="observed")
scatter!(E, sum(ass.bkg_EH[1], dims=2)[:] .+ sum(ass.Npred_EH_oscs), label="bg + pred oscs")

Legend(f[1, 2],ax)

ax = Axis(f[2, 1])

scatter!(E, (sum(ass.bkg_EH[1], dims=2)[:] .+ sum(ass.Npred_EH_oscs)) ./ ass.observed[1])
f
```

```julia
Revise.revise()
```

```julia
spectrum = CSV.read("../../../physics/daya_bay_flux.csv", DataFrame, header=1, delim="  ");
bin_edges = sort(union(spectrum[!, 1], spectrum[!, 2]))
flux = spectrum[!, 3]
flux_errs = spectrum[!, 4]
binc = bin_edges[1:end-1] + diff(bin_edges) / 2
```

```julia
scatter(binc, flux)
```

```julia
res = CSV.read("response_matrix.txt", DataFrame, delim="\t", skipto=6);
```

```julia
res = Matrix(res[!, 1:end-1]); # should be normalised s.t. prompt energy is conserved
```

```julia
res = float.(res);
```

```julia
for (c, v) in enumerate(eachrow(res))
    # println(c, v)
    if any(isnan, v) || iszero(v)
        res[c, :] .= zeros(size(v))
    else
        res[c, :] .= res[c, :] ./ sum(v)
    end
end

```

```julia
E_antinu = Vector(range(1, 13; step=0.01))
E_prompt = Vector(range(1-0.05, 8; step=0.05))
E_prompt[1] = 0.7;
```

```julia
#f = Figure()
#ax = Axis(f[1, 1], xlabel="anti nu energy [MeV]", ylabel="prompt energy [MeV]")
fig, ax, hm = heatmap(E_antinu, E_prompt, res)
Colorbar(fig[:, end+1], hm)
fig
```

```julia
# put flux through the detector response
```

```julia
E_antinu_binc = (E_antinu[1:end-1] + E_antinu[2:end]) / 2;
E_prompt_binc = (E_prompt[1:end-1] + E_prompt[2:end]) / 2;
```

```julia
reactor = Newtrinos.reactor_flux.configure()
```

```julia
pulls = ones(25)

sys = [reactor.sys_flux(e, pulls) for e in E_antinu_binc]
reactor.nominal_flux.(E_antinu_binc) .+ sys
```

```julia
db = Newtrinos.dayabay_rewrite.configure()
```

```julia
db.assets
```

```julia
round.(Int, [2.2])
```

```julia
experiments = (dayabay = Newtrinos.dayabay_rewrite.configure(),)

vars_to_scan = OrderedDict()
vars_to_scan[:θ₁₃] = 31
vars_to_scan[:Δm²₃₁] = 31

likelihood = Newtrinos.generate_likelihood(experiments);

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
```

```julia
db = experiments.dayabay.assets
```

```julia
period = "Six"
EH = "3"


"dfBKG_$(period)_EH$EH"
```

```julia
names(db.dfBKG_dict["dfBKG_Six_EH3"])
```

```julia
f = Figure()
ax = Axis(f[1, 1])
plot!(db.E_arrs[1], db.baseline_av_best_fit_prob_arr[1])
f
```

```julia
db.dfBKG_dict["dfBKG_Eight_EH1"]
```

```julia
predicted_no_osc = experiments.dayabay.assets.predicted_no_oscs
official_no_osc = experiments.dayabay.assets.Npred_EH3_nooscs
energy_bins = experiments.dayabay.assets.energy_bins
e_binc = (energy_bins[2:end] + energy_bins[1:end-1]) / 2;
```

```julia
f = Figure()
ax = Axis(f[1, 1])
scatter!(e_binc, sum(official_no_osc))
scatter!(e_binc, sum(predicted_no_osc))
f
```

```julia
logdensityof(likelihood, p)
```

```julia
mle = Newtrinos.find_mle(likelihood, distprod(priors), p)
```

```julia
best_fitting_params = mle[3]
```

```julia
asimov_mle = Newtrinos.generate_asimov_data(experiments.dayabay, best_fitting_params)
```

```julia
experiments.dayabay.plot(best_fitting_params)
```

```julia
f = Figure()
ax = Axis(f[1, 1])
scatter!(e_binc, asimov_mle)
scatter!(e_binc, experiments.dayabay.assets.observed)

f
```

```julia
delta_m32 = results.Δm²₃₁ - results.Δm²₂₁
sin22theta13 = sin(2 * results.θ₁₃) ^ 2

delta_m32, sin22theta13
```

```julia
sim = Newtrinos.generate_asimov_data(experiments.dayabay, p)
```

```julia
p.pulls .= 1
```

```julia
Newtrinos.dayabay_rewrite.get_expected_per_period(p, 1, db.physics, db.assets)
```

```julia
scatter(e_binc, sim)
```

```julia
experiments.dayabay.assets.energy_bins
```

```julia
experiments.dayabay.assets.energy_resolution
```

```julia
exp = Newtrinos.dayabay_rewrite.get_expected(p, experiments.dayabay.physics, experiments.dayabay.assets) * experiments.dayabay.assets.norm
```

```julia

```

```julia
scatter(e_binc, exp)
```

```julia
vec(zeros(1, 20))
```

```julia
experiments.dayabay.assets.energy_bins
```

```julia
db_nom = Newtrinos.reactor_flux.DayaBayFlux()
db_sys = Newtrinos.reactor_flux.DayaBaySystematics()
```

```julia
p = Newtrinos.get_params(experiments)
```

```julia
Newtrinos.reactor_flux.get_params(db_nom)
```

```julia
physics = Newtrinos.dayabay_rewrite.default_physics()
assets = Newtrinos.dayabay_rewrite.get_assets(physics)
```

```julia
period = 1
E = assets.E_antinu_binc
L = assets.L_arrs[period]
weights = assets.flux_weights_bar_osc[period]
prob_arr = physics.osc.osc_prob(E, L, p, anti=true)[:, :, 1, 1]'
```

```julia
for i in eachindex(weights)
    prob_arr[i, :] .*= weights[i]
end
```

```julia
sum(prob_arr, dims=1)
```

```julia

```

```julia

```

```julia

```

```julia
experiments.dayabay.assets
```

```julia
physics = Newtrinos.dayabay_rewrite.default_physics()
```

```julia
physics.flux.nominal_flux(6)
```

```julia
predicted_no_osc = experiments.dayabay.assets.predicted_no_oscs
official_no_osc = experiments.dayabay.assets.Npred_EH3_nooscs
energy_bins = experiments.dayabay.assets.energy_bins
e_binc = (energy_bins[2:end] + energy_bins[1:end-1]) / 2;
```

```julia
sum(sum(predicted_no_osc))
```

```julia
sum(sum(official_no_osc))
```

```julia
predicted_no_osc
```

```julia
official_no_osc
```

```julia
norm = sum(sum(stack(official_no_osc), dims=1)) / sum(sum(stack(predicted_no_osc), dims=1))
```

```julia
norm
```

```julia
sum(stack(official_no_osc), dims=1)
```

```julia
for i in eachindex(predicted_no_osc)
    predicted_no_osc[i] .*= norm
end
```

```julia
sum(predicted_no_osc[1]), sum(official_no_osc[1])
```

```julia
markersize = 10


f = Figure()
ax = Axis(f[1, 1], xlabel="reconstructed prompt energy [MeV]",)
scatter!(e_binc, official_no_osc[1], color="blue", markersize=markersize, marker=:utriangle)
scatter!(e_binc, official_no_osc[2], color="red", markersize=markersize, marker=:utriangle)
scatter!(e_binc, official_no_osc[3], color="green", markersize=markersize, marker=:utriangle)
scatter!(e_binc, sum(official_no_osc), color="black", markersize=markersize, marker=:utriangle, label="DayaBay")
scatter!(e_binc, predicted_no_osc[1], color="blue", markersize=markersize, marker=:xcross)
scatter!(e_binc, predicted_no_osc[2], color="red", markersize=markersize, marker=:xcross)
scatter!(e_binc, predicted_no_osc[3], color="green", markersize=markersize, marker=:xcross)
scatter!(e_binc, sum(predicted_no_osc), color="black", markersize=markersize, marker=:xcross, label="Newtrinos.jl")
axislegend(ax)
f
#save("daya_bay_prediction.png", f)
```

```julia
f = Figure()
ax = Axis(f[1, 1], xlabel="reconstructed prompt energy [MeV]")
scatter!(e_binc, predicted_no_osc[1] ./ official_no_osc[1])
scatter!(e_binc, predicted_no_osc[2] ./ official_no_osc[2])
scatter!(e_binc, predicted_no_osc[3] ./ official_no_osc[3])
scatter!(e_binc, sum(predicted_no_osc) ./ sum(official_no_osc), color="black", markersize=3)
f
```

```julia
f = Figure()
ax = Axis(f[1, 1], xlabel="reconstructed prompt energy [MeV]")
scatter!(e_binc, predicted_no_osc[1] ./ official_no_osc[1])
scatter!(e_binc, predicted_no_osc[2] ./ official_no_osc[2])
scatter!(e_binc, predicted_no_osc[3] ./ official_no_osc[3])
f
```

```julia
predicted_no_osc, official_no_osc
```

```julia
show(err)
```

```julia
db_bin_edges = experiments.dayabay.assets.energy_bins
db_binc = (db_bin_edges[1:end-1] + db_bin_edges[2:end]) / 2;
```

```julia

ibd_weighted_flux = huber.(E_antinu_binc) .* xsec.(E_antinu_binc)
out = transpose(res) * ibd_weighted_flux
idx = searchsortedlast.(Ref(db_bin_edges), E_prompt_binc)
println(size(idx))
result = zeros(size(db_binc))
# integrate by summing over Eprompt in the bins of Dayabay
for c in eachindex(db_binc)
    result[c] = sum(out[idx .== c])
end

```

```julia
scatter(db_binc, result)
```

```julia
# Reactors / Baselines, etc
exp_dict = Dict("Detector" => ["AD1", "AD2", "AD3", "AD8", "AD4", "AD5", "AD6", "AD7"],
    "EH" => ["EH1", "EH1", "EH2", "EH2", "EH3", "EH3", "EH3", "EH3"],
    "Target [kg]" => [19941, 19967, 19891, 19944, 19917, 19989, 19892, 19931],
    "Efficiency" => [0.7743, 0.7716, 0.8127, 0.8105, 0.9513, 0.9514, 0.9512, 0.9513],
    "Six-AD Period" => [true, true, true, false, true, true, true, false],
    "Eight-AD Period" => [true, true, true, true, true, true, true, true],
    "Seven-AD Period" => [true, false, true, true, true, true, true, true],
    "D1" => [362.38, 357.94, 1332.48, 1337.43, 1919.63, 1917.52, 1925.26, 1923.15],
    "D2" => [371.76, 368.41, 1358.15, 1362.88, 1894.34, 1891.98, 1899.86, 1897.51],
    "L1" => [903.47, 903.35, 467.57, 472.97, 1533.18, 1534.92, 1538.93, 1540.67],
    "L2" => [817.16, 816.90, 489.58, 495.35, 1533.63, 1535.03, 1539.47, 1540.87],
    "L3" => [1353.62, 1354.23, 557.58, 558.71, 1551.38, 1554.77, 1556.34, 1559.72],
    "L4" => [1265.32, 1265.89, 499.21, 501.07, 1524.94, 1528.05, 1530.08, 1533.18])
df_exp = DataFrame(exp_dict);
```

```julia
exp_dict["Efficiency"]
```

```julia
# flux weight is power / (4 * pi * d^2)
# total flux is sum over all reactors,
# need to evaluate separately for all ADs bc of different baselines for all reactor-AD pairs
# so, evaluate on a grid? yes!
# make grid on AD x Reactor baseline, get P_surv for all energies on this grid, add to flux per AD
```

```julia
osc = Newtrinos.osc.configure()
```

```julia
flux = huber.(E_antinu_binc);
```

```julia
EH_list = [1, 2, 3]

# Data-taking Periods
period_list = ["Six", "Eight", "Seven"]
period_idx = [20, 49, 78]
periods_dict = Dict("Six" => 6, "Eight" => 8, "Seven" => 7)

# Reactors
reactor_list = ["D1", "D2", "L1", "L2", "L3", "L4"]
```

```julia
df_exp[!, "D1"]
```

```julia
df_EH3 = filter(row -> row["EH"] == "EH3", df_exp)
L_matrix = Matrix(df_EH3[:, ["D1", "D2", "L1", "L2", "L3", "L4"]])
```

```julia
ad_list = df_EH3[!, "Detector"]
```

```julia
length(df_EH3[!, :Detector])
```

```julia
flux = huber.(E_antinu_binc)

# Make some flux matrix
flux_dict = Dict("D1" => copy(flux), "D2" => copy(flux), "L1" => copy(flux), "L2" => copy(flux), "L3" => copy(flux), "L4" => copy(flux))
df_flux = DataFrame(flux_dict);
```

```julia
df_flux[!, :D1]
```

```julia
flux_matrix = Matrix(df_flux)   # energy x reactor
```

```julia
flux_matrix
```

```julia
flux
```

```julia
bestift_osc = Newtrinos.osc.configure()
@reset p.θ₁₂ = asin(sqrt(0.307))
@reset p.θ₁₃ = asin(sqrt(0.0851)) * 0.5
@reset p.θ₂₃ = asin(sqrt(0.57))
@reset p.δCP = 0.
@reset p.Δm²₂₁ = 7.53e-5
@reset p.Δm²₃₁ = 2.466e-3 + p[:Δm²₂₁]
```

```julia
#print(L_matrix)   # has dimension AD x reactor
#println()
cross_sec = xsec.(E_antinu_binc)
total_flux = zeros(size(E_antinu_binc))

flux_all_reactors = stack([flux for i in eachindex(reactor_list)])
for period in period_list
    df_period = filter(row -> row["$(period)-AD Period"], df_exp)
    # takes only data of EH3 (furthest)
    df_period = filter(row -> row["EH"] == "EH3", df_period)
    #println(df_period)
    L_matrix = Matrix(df_period[:, ["D1", "D2", "L1", "L2", "L3", "L4"]])
    println(size(L_matrix))
    # AD x reactor
    L_arr = vec(L_matrix)
    # get weighted flux
    #flux .* df_period[:, "Target [kg]"] .* df_period[:, "Efficiency"]
    println(size(L_matrix))
    # (AD x reactor).flatten() x energy
    best_fit_prob_arr = bestift_osc.osc_prob(E_antinu_binc, L_arr, p, anti=true)[:, :, 1, 1]'
    println(size(best_fit_prob_arr))
    #println(df_period[!, :Detector])
    efficiency = df_period[!, "Efficiency"]
    target_mass = df_period[!, "Target [kg]"]
    #print(size(efficiency))
    for c_ad in eachindex(df_period[!, "Detector"])
        ad_flux = zeros(size(E_antinu_binc))
        for (c_reac, reactor) in enumerate(reactor_list)
            ad_flux .+= flux ./ L_matrix[c_ad, c_reac]^2
        end
        ad_flux .*= efficiency[c_ad] * target_mass[c_ad]
        total_flux .+= ad_flux

        
    end
end

ibd_weighted_flux = total_flux .* cross_sec
out = transpose(res) * ibd_weighted_flux
```

```julia
period = period_list[1]
df_period = filter(row -> row["$(period)-AD Period"], df_exp)
# takes only data of EH3 (furthest)
df_period = filter(row -> row["EH"] == "EH3", df_period)
```

```julia
L_matrix = df_period[:, ["D1", "D2", "L1", "L2", "L3", "L4"]]
# indices: AD x reactor
L_arr = vec(Matrix(L_matrix))
```

```julia
eff

```

```julia
efficiency = repeat(eff, 6)
target_mass = repeat(mass, 6);
```

```julia
L_arr.^2 .* efficiency .*target_mass
```

```julia
L_arr
```

```julia
stack([flux for i in eachindex(reactor_list)])
```

```julia
idx = searchsortedlast.(Ref(db_bin_edges), E_prompt_binc)
println(size(idx))
result = zeros(size(db_binc))
# integrate by summing over Eprompt in the bins of Dayabay
for c in eachindex(db_binc)
    result[c] = sum(out[idx .== c])
end
```

```julia

scatter(db_binc, result)
```

```julia
result[end]
```

```julia

```

```julia

```

```julia
mle = Newtrinos.find_mle(likelihood, distprod(;priors...), p)
```

```julia
mle
```

```julia

```
