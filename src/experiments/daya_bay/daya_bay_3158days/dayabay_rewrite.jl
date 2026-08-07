module dayabay_rewrite
using DataFrames
using CSV
using LinearAlgebra
using Distributions
using DataStructures
using CairoMakie
using Accessors
using Logging
using BAT
import ..Newtrinos

@kwdef struct DayaBay <: Newtrinos.Experiment
    physics::NamedTuple
    params::NamedTuple
    priors::NamedTuple
    assets::NamedTuple
    forward_model::Function
    plot::Function
end

function default_physics()
    osc = Newtrinos.osc.configure()
    xsec = Newtrinos.ibd_xsec.configure()
    flux = Newtrinos.reactor_flux.configure()
    (; osc, xsec, flux)
end

function configure(physics=default_physics())
    physics = (;physics.osc, physics.xsec, physics.flux)
    assets = get_assets(physics)
    return DayaBay(
        physics = physics,
        params = get_params(),
        priors = get_priors(),
        assets = assets,
        forward_model = get_forward_model(physics, assets),
        plot = get_plot(physics, assets)
    )
end

function get_params()
    # (background_norm = ones(5),)
    return (norm=1.,)
end


function get_priors()
    exp = ones(5)
    cv = Diagonal(ones(5))
    priors = (
        background_norm = Distributions.MvNormal(exp, cv),
    )
    #return priors
    return (norm = Distributions.Uniform(0.5, 1.5),)
end

function get_assets(physics; datadir = @__DIR__)

    @info "Loading dayabay data"
    # Experimental Halls
    EH_list = [1, 2, 3]
    
    # Data-taking Periods
    period_list = ["Six", "Eight", "Seven"]
    period_idx = [20, 49, 78]
    periods_dict = Dict("Six" => 6, "Eight" => 8, "Seven" => 7)    # must...resist....
    
    # Reactors
    reactor_list = ["D1", "D2", "L1", "L2", "L3", "L4"]
    
    # Correlation matrix from Fig. 29 in https://arxiv.org/pdf/1607.05378.pdf
    corr_mat_df = CSV.read(joinpath(datadir, "DayaBay_CorrMat_arXiv_1607.05378.txt"), DataFrame, delim=' ')
    corr_mat = Symmetric(Matrix(corr_mat_df[:, 5:end]))
    
    # Fractional systematic uncertainty on the expected IBD energy spectra
    rel_unc_diag = corr_mat_df.diag_sys_unc_relative_to_spectrum;

    # from https://arxiv.org/pdf/2211.14988
    lifetime = Dict("Six-AD Period" => 217, "Eight-AD Period" => 1524, "Seven-AD Period" => 1417)
    
    # Reactors / Baselines, etc
    exp_dict = Dict("Detector" => ["AD1", "AD2", "AD3", "AD8", "AD4", "AD5", "AD6", "AD7"],    # antinu-detectors
                "EH" => ["EH1", "EH1", "EH2", "EH2", "EH3", "EH3", "EH3", "EH3"],   # exp hall
                "Target [kg]" => [19941, 19967, 19891, 19944, 19917, 19989, 19892, 19931],   # to scale linearly expected events
                "Efficiency" => [0.7743, 0.7716, 0.8127, 0.8105, 0.9513, 0.9514, 0.9512, 0.9513],  # to scale linearly expected events
                "Six-AD Period" => [true, true, true, false, true, true, true, false],   # online time
                "Eight-AD Period" => [true, true, true, true, true, true, true, true],
                "Seven-AD Period" => [true, false, true, true, true, true, true, true],
                "D1" => [362.38, 357.94, 1332.48, 1337.43, 1919.63, 1917.52, 1925.26, 1923.15],   # baselines to all reactors
                "D2" => [371.76, 368.41, 1358.15, 1362.88, 1894.34, 1891.98, 1899.86, 1897.51],   # D: Daya Bay
                "L1" => [903.47, 903.35, 467.57, 472.97, 1533.18, 1534.92, 1538.93, 1540.67],     # L: the other one...
                "L2" => [817.16, 816.90, 489.58, 495.35, 1533.63, 1535.03, 1539.47, 1540.87],
                "L3" => [1353.62, 1354.23, 557.58, 558.71, 1551.38, 1554.77, 1556.34, 1559.72],
                "L4" => [1265.32, 1265.89, 499.21, 501.07, 1524.94, 1528.05, 1530.08, 1533.18])
    df_exp = DataFrame(exp_dict);
    
    function parse(fname, idx, len)
        # Parsing the DayaBay format CSV files
        header = Array(CSV.read(fname, DataFrame, delim=' ', skipto=idx-1, limit=1, ignorerepeated=true, header=false));
        header = header[2:end];
        df = CSV.read(fname, DataFrame, delim=' ', skipto=idx, limit=len, ignorerepeated=true, header=false);
        rename!(df, header, makeunique=false);
        return df
    end
    
    # Dicts to fill
    dfBKG_dict = Dict()
    dfIBD_dict = Dict()

    # Tell me, do we really need this?
    bkg_types = ["Nacc", "Nalphan", "Namc", "Nlihe", "Nfastn"]
    
    # Parse IBD and background files
    for EH in EH_list
        fileBKG = joinpath(datadir,  "DayaBay_BackgroundSpectrum_EH$(EH)_3158days.txt")
        fileIBD = joinpath(datadir,  "DayaBay_IBDPromptSpectrum_EH$(EH)_3158days.txt")
        dfIBD_dict["dfIBD_EH$EH"] = parse(fileIBD, 11, size(corr_mat_df, 1))
        for i in 1:length(period_list)
            period = period_list[i]
            idx = period_idx[i]
            dfBKG_dict["dfBKG_$(period)_EH$(EH)"] = parse(fileBKG, idx, size(corr_mat_df, 1))
        end
        # Sum over data-taking periods
        dfIBD_dict["dfIBD_EH$EH"][!, "Nobs"] = sum(eachcol(dfIBD_dict["dfIBD_EH$EH"][!, ["Nobs_6AD", "Nobs_8AD", "Nobs_7AD"]]))
        dfIBD_dict["dfIBD_EH$EH"][!, "Npred"] = sum(eachcol(dfIBD_dict["dfIBD_EH$EH"][!, ["Npred_6AD", "Npred_8AD", "Npred_7AD"]]))
        BKG = zeros(size(dfIBD_dict["dfIBD_EH$EH"], 1))
        for period in period_list
            BKG .+= dfBKG_dict["dfBKG_$(period)_EH$(EH)"].Nbkg
        end
        dfIBD_dict["dfIBD_EH$EH"][!, "BKG"] = BKG
        dfIBD_dict["dfIBD_EH$EH"][!, "N"] = dfIBD_dict["dfIBD_EH$EH"].Nobs .- BKG
    end

    # Extract all background templates
    bkg_templates = Dict()
    for EH in EH_list
        for period in period_list
            bkg_templates["$(period)_$(EH)"] = Matrix(dfBKG_dict["dfBKG_$(period)_EH$(EH)"][!, [:Nacc, :Nalphan, :Namc, :Nlihe, :Nfastn]])
        end
    end

    # For now, sum background templates for each EH over the periods
    bkg_EH = Dict()
    for EH in EH_list
        bkg = []
        for period in period_list
            push!(bkg, bkg_templates["$(period)_$(EH)"])
        end
        bkg_EH[EH] = sum(bkg, dims=1)[1]
    end

    energy_bins = copy(dfBKG_dict["dfBKG_Six_EH3"].Emin)
    energy = copy(dfBKG_dict["dfBKG_Six_EH3"].Ec)
    push!(energy_bins, dfBKG_dict["dfBKG_Six_EH3"].Emax[end])
    
    # Setup for DayaBay bestfit to recalculate unoscillated spectrum
    bestift_osc = Newtrinos.osc.configure()
    @reset bestift_osc.params.θ₁₂ = asin(sqrt(0.307))
    @reset bestift_osc.params.θ₁₃ = asin(sqrt(0.0851)) * 0.5
    @reset bestift_osc.params.θ₂₃ = asin(sqrt(0.57))
    @reset bestift_osc.params.δCP = 0.
    @reset bestift_osc.params.Δm²₂₁ = 7.53e-5
    @reset bestift_osc.params.Δm²₃₁ = 2.466e-3 + bestift_osc.params[:Δm²₂₁]
    
    
    #maps anti nu energy to prompt energy
    res = CSV.read("response_matrix.txt", DataFrame, delim="\t", skipto=6)
    res = Matrix(res[!, 1:end-1])
    res = float.(res);
    
    #normalise s.t. we can use this as a proper smearing matrix
    for (c, v) in enumerate(eachrow(res))
        # println(c, v)
        # avoid dividing by zero
        if any(isnan, v) || iszero(v)
            res[c, :] .= zeros(size(v))
        else
            res[c, :] .= res[c, :] ./ sum(v)
        end
    end
    
    energy_resolution = transpose(res)
    
    #bin edges
    E_antinu = Vector(range(1, 13; step=0.01))
    E_prompt = Vector(range(1-0.05, 8; step=0.05))
    E_prompt[1] = 0.7
    
    # bin centers
    E_antinu_binc = (E_antinu[1:end-1] + E_antinu[2:end]) / 2;
    E_prompt_binc = (E_prompt[1:end-1] + E_prompt[2:end]) / 2;
    
    # flux and ibd cross section
    flux = physics.flux.nominal_flux
    xsec = physics.xsec.xsec
    
    flux_eval = flux.(E_antinu_binc)
    xsec_eval = xsec.(E_antinu_binc)
    ibd_weighted_flux = flux_eval .* xsec_eval
    
    predicted_no_oscs = Vector{Vector{Vector{Float64}}}()
    flux_weights_bar_osc = Vector{Vector{Vector{Float64}}}()
    baseline_av_best_fit_prob_arr = Vector{Vector{Float64}}()
    
    E_arrs = Vector{Vector{Float64}}()
    L_arrs = Vector{Vector{Vector{Float64}}}()
    Npred_EH_nooscs = Vector{Vector{Vector{Float64}}}()
    Npred_EH_oscs = Vector{Vector{Vector{Float64}}}()

    efficiency = Vector{Vector{Vector{Float64}}}()

    norm = []

    for EH in EH_list
        E_arr = dfIBD_dict["dfIBD_EH$(EH)"].Ec .+ 0.78
        # indices: AD x reactor
        push!(L_arrs, Vector{Vector{Float64}}())
        push!(flux_weights_bar_osc, Vector{Vector{Float64}}())
        push!(Npred_EH_nooscs, Vector{Vector{Float64}}())
        push!(Npred_EH_oscs, Vector{Vector{Float64}}())
        push!(predicted_no_oscs, Vector{Vector{Float64}}())
        #push!(efficiency, Vector{Vector{Float64}}())
        for period in period_list
            df_period = filter(row -> row["$(period)-AD Period"], df_exp)
            # takes only data of EH3 (furthest)
            df_period = filter(row -> row["EH"] == "EH$(EH)", df_period)
            # shape depends on period, as #AD differs
            L_matrix = df_period[:, ["D1", "D2", "L1", "L2", "L3", "L4"]]
            # flatten
            L_arr = vec(Matrix(L_matrix))
            push!(E_arrs, E_arr)
            push!(L_arrs[end], L_arr)
            # for comparison with proper prediction, keep this
            Npred_EH_after_best_fit_osc = dfIBD_dict["dfIBD_EH$(EH)"][:, "Npred_$(periods_dict[period])AD"]
            best_fit_prob_arr = bestift_osc.osc_prob(E_arr, L_arr, bestift_osc.params, anti=true)[:, :, 1, 1]'
            # get the survival probability as flux scale, scale flux by respective baseline squared, normalise by inverse squared baselines
            baseline_average_best_fit_prob_arr = vec(sum(best_fit_prob_arr ./ (L_arr .^ 2), dims=1) ./ sum(1 ./(L_arr .^ 2)))
            push!(baseline_av_best_fit_prob_arr, baseline_average_best_fit_prob_arr)
            # unoscillated N predicted EH3:
            Npred_EH_noosc = Npred_EH_after_best_fit_osc ./ baseline_average_best_fit_prob_arr
            # TODO: ONLY REPLACE THIS HERE! DO NOT CHANGE ANYTHING ELSE
            # tis but the flux / (4*pi*L^2) * IBD, put through the detector response, multiplied by respective efficiency and target mass

            # Npred_EH3_noosc_from_flux = 
            push!(Npred_EH_nooscs[end], Npred_EH_noosc)
            push!(Npred_EH_oscs[end], Npred_EH_after_best_fit_osc)
            
            eff = df_period[:, "Efficiency"]
            mass = df_period[:, "Target [kg]"]
            efficiency = repeat(eff, length(reactor_list))
            target_mass = repeat(mass, length(reactor_list));

            # collects the weights with which the anti-nu flux has to be multiplied
            # dimensions: period x (#AD x reactor).flatten latter two stem from L_arr
            flux_weight_bar_osc = lifetime["$period-AD Period"] * efficiency .* target_mass ./ L_arr.^2 / sum(1 ./ L_arr.^2)
            push!(flux_weights_bar_osc[end], flux_weight_bar_osc)
            smeared = energy_resolution * ibd_weighted_flux .* sum(flux_weight_bar_osc)
            idx = searchsortedlast.(Ref(energy_bins), E_prompt_binc)
            result = zeros(size(energy_bins[1:end-1]))
            # integrate by summing over Eprompt in the bins of Dayabay
            for c in eachindex(energy_bins[1:end-1])
                result[c] = sum(smeared[idx .== c])
            end

            push!(predicted_no_oscs[end], result)
        end
        #calculate per EH norm of forward-modelled counts
        push!(norm, sum(sum(Npred_EH_oscs[end])) / sum(sum(predicted_no_oscs[end])))
    end
    
    observed = []
    for EH in EH_list
        push!(observed, round.(Int, dfIBD_dict["dfIBD_EH$(EH)"].Nobs))
    end
    # observed = 
    #observed = round.(Int, dfIBD_dict["dfIBD_EH1"].N)
    observed = reshape(stack(observed), 26*3) 
    #for i in eachindex(predicted_no_oscs)
    #    predicted_no_oscs[i] .*= norm
    #end

    nom_flux       = physics.flux.nominal_flux.(E_antinu_binc)
    prompt_bin_idx = searchsortedlast.(Ref(energy_bins), E_prompt_binc)

    assets = (;
        E_arrs, 
        L_arrs, 
        Npred_EH_nooscs,
        Npred_EH_oscs,
        observed,
        energy_bins,
        period_list,
        EH_list,
        energy_resolution,
        predicted_no_oscs,
        xsec_eval,     
        E_antinu_binc,
        E_antinu,
        E_prompt_binc,
        E_prompt,   
        flux_weights_bar_osc,
        norm,
        energy,
        dfBKG_dict,
        dfIBD_dict,
        baseline_av_best_fit_prob_arr,
        bkg_templates,
        bkg_EH,
        nom_flux,
        prompt_bin_idx,
    )

end


function get_expected_per_EH(params, EH, physics, assets)
    E_antinu_binc = assets.E_antinu_binc
    E_prompt_binc = assets.E_prompt_binc
    E_prompt = assets.E_prompt
    energy_bins = assets.energy_bins
    periods = assets.period_list
    energy_resolution = assets.energy_resolution
    osc_prob = physics.osc.osc_prob
    pulls = params.pulls
    T = eltype(pulls)
    xsec_eval = assets.xsec_eval
    p_ibd_weighted_flux = zeros(T, (length(periods), length(E_antinu_binc)))
    L_arrs = assets.L_arrs[EH]
    flux_weights = assets.flux_weights_bar_osc[EH]
    flux = physics.flux
    nom_flux = assets.nom_flux

    sys_flux  = [flux.sys_flux(e, pulls) for e in E_antinu_binc]
    flux_eval = assets.nom_flux .+ sys_flux

    osc_weighted = zeros(T, length(E_antinu_binc))
    for c in eachindex(periods)
        S = @view osc_prob(E_antinu_binc, L_arrs[c], params, anti=true)[:, :, 1, 1]
        mul!(osc_weighted, S, flux_weights[c], 1, 1)   # osc_weighted += S * w
    end
    weighted  = @. flux_eval * assets.xsec_eval * osc_weighted

    smeared = assets.energy_resolution * weighted
    #println(size(smeared))
    idx    = assets.prompt_bin_idx
    nbins  = length(assets.energy_bins) - 1
    result = zeros(T, nbins)
    @inbounds for k in eachindex(idx)
        c = idx[k]
        (1 <= c <= nbins) && (result[c] += smeared[k])
    end

    result .*= assets.norm[EH] * params.norm

    return result
end


function get_expected(params, physics, assets)
    reshape(stack([get_expected_per_EH(params, EH, physics, assets) for EH in assets.EH_list]), 26 * 3)
end

function get_forward_model(physics, assets)
    function forward_model(params)
        exp_events = get_expected(params, physics, assets)
        distprod(Poisson.(exp_events))
    end
end

function get_plot(physics, assets)

    function plot(params, data=assets.observed)
        
        m = mean(get_forward_model(physics, assets)(params))
        v = var(get_forward_model(physics, assets)(params))
    
        f = Figure()
        ax = Axis(f[1,1])
        
        plot!(ax, assets.energy, data, color=:black, label="Observed")
        stephist!(ax, assets.energy, weights=m, bins=assets.energy_bins, label="Expected")
        barplot!(ax, assets.energy, m .+ sqrt.(v), width=diff(assets.energy_bins), gap=0, fillto= m .- sqrt.(v), alpha=0.5, label="Standard Deviation")
        
        ax.ylabel="Counts"
        ax.title="Daya Bay"
        axislegend(ax, framevisible = false)
        
        
        ax2 = Axis(f[2,1])
        plot!(ax2, assets.energy, data ./ m, color=:black, label="Observed")
        hlines!(ax2, 1, label="Expected")
        barplot!(ax2, assets.energy, 1 .+ sqrt.(v) ./ m, width=diff(assets.energy_bins), gap=0, fillto= 1 .- sqrt.(v)./m, alpha=0.5, label="Standard Deviation")
        ylims!(ax2, 0.9, 1.1)
        
        ax.xticksvisible = false
        ax.xticklabelsvisible = false
        
        rowsize!(f.layout, 1, Relative(3/4))
        rowgap!(f.layout, 1, 0)
        
        ax2.xlabel="Eₚ (MeV)"
        ax2.ylabel="Counts/Expected"
    
        xlims!(ax, minimum(assets.energy_bins), maximum(assets.energy_bins))
        xlims!(ax2, minimum(assets.energy_bins), maximum(assets.energy_bins))
        
        ylims!(ax, 0, 60000)
        
        f
    
    end
end

end
