using Distributions
using DensityInterface
using BAT
using DataStructures
using Newtrinos
using LinearAlgebra
using DelimitedFiles
using FileIO
using Accessors
using CairoMakie
using DataFrames
using CSV


results = Dict()
for (c, ordering) in enumerate([:NO, :IO])

    osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(
        interaction = Newtrinos.osc.SI(),
        eigen_method = Newtrinos.BargerEigen(),
        flavour=Newtrinos.osc.ThreeFlavour(ordering=ordering)
        ))
    xsec = Newtrinos.ibd_xsec.configure(Newtrinos.ibd_xsec.VogelBeacomO1())
    flux = Newtrinos.dayabay_flux.configure(Newtrinos.dayabay_flux.ReactorFluxConfig(average=true, N_pulls=25))
    physics = (; osc, xsec, flux)
    experiments = (juno=Newtrinos.juno_rewrite.configure(physics), tao=Newtrinos.tao_rewrite.configure(physics))

    p = Newtrinos.get_params(experiments)
    priors = Newtrinos.get_priors(experiments)
    likelihood = Newtrinos.generate_likelihood(experiments)

    val = 0.0851
    err = 0.0024
    theta13 = asin(sqrt(val)) / 2
    theta13_sigma = 1 / (4 * sqrt(-(val - 1) * val)) * err
    spec_pulls_cov = FileIO.load("spectrum_pulls_cov_25.jld2")["cov"]
    spec_pulls_mle = FileIO.load("spectrum_pulls_cov_25.jld2")["mle"]
    temp = UpperTriangular(spec_pulls_cov)
    temp += transpose(temp) - Diagonal(diag(temp))
    @reset priors.spectrum_pulls = MvNormal(spec_pulls_mle, temp)
    @reset p.spectrum_pulls = spec_pulls_mle

    @reset priors.θ₂₃ = p.θ₂₃
    @reset priors.δCP = p.δCP 
    @reset priors.Δm²₂₁ = Uniform(7.1e-5, 7.9e-5)
    @reset priors.θ₁₂ = Uniform(asin(sqrt(0.275)), asin(sqrt(0.32)))
    @reset priors.θ₁₃ = Normal(theta13, theta13_sigma)
    @reset p.Δm²₃₁ = 2.5413e-3
    @reset p.θ₁₃ = theta13
    @reset p.Δm²₂₁ = 7.388e-5
    @reset p.θ₁₂ = asin(sqrt(0.3036))
    low = 1.8e-3
    high = 3.0e-3
    if ordering == :NO
        @reset priors.Δm²₃₁ = Uniform(low, high)
    else
        @reset priors.Δm²₃₁ = Uniform(-high, -low)
    end
    if !isdir("combined_output")
        mkdir("combined_output")
    end

    vars_to_scan = OrderedDict()
    vars_to_scan[:Δm²₃₁] = 120
    result = Newtrinos.profile(likelihood, priors, vars_to_scan, p, cache_dir="combined_scratch_$(String(ordering))")
    FileIO.save("combined_output/test_dm31_juno_tao_$(String(ordering)).jld2", Dict("result" => result))
    converted = Newtrinos.NewtrinosResult(axes = (delta2m31= result.axes.Δm²₃₁,), values=result.values);
    results[ordering] = converted
end

f = Figure()
ax = Axis(f[1, 1], )


IO_off = readdlm("207_days/fig_4_chi2.txt", skipstart=25)
IO_off = convert.(Float64, IO_off[IO_off[:, 1] .== "IO", 2:3])


NO_off = readdlm("207_days/fig_4_chi2.txt", skipstart=25)
NO_off = convert.(Float64, NO_off[NO_off[:, 1] .== "NO", 2:3]);


dayabay_NO = Normal(2.5413e-3, 0.060e-3)
dayabay_IO = Normal(-2.4957e-3, 0.060e-3)
base = minimum([-maximum(results[:IO].values.log_posterior), -maximum(results[:NO].values.log_posterior)])

f = Figure()
ax_IO = Axis(f[1, 1], xlabel="Δm²₃₁ [eV²]", ylabel="-2 * ΔLLH", xminorticks=  IntervalsBetween(5), title="IO", yminorticksvisible = true, yticks=0:5:30)
ax_NO = Axis(f[1, 2], xlabel="Δm²₃₁ [eV²]", xminorticks=  IntervalsBetween(5), title="NO", yminorticksvisible = true, yticks=0:5:30,yaxisposition = :right)

axes = [ax_IO, ax_NO]
lines!(ax_IO, results[:IO].axes.delta2m31, 2 .* (-results[:IO].values.log_posterior .- base), label="Newtrinos.jl")
lines!(ax_NO, results[:NO].axes.delta2m31, 2 .* (-results[:NO].values.log_posterior .- base))

for (a, db, res) in zip(axes, [dayabay_IO, dayabay_NO], [results[:IO], results[:NO]])
    x = collect(LinRange(res.axes.delta2m31[1], res.axes.delta2m31[end], 1000))
 
    lines!(a, x, 2 * (-logdensityof.(db, x) .+ logdensityof(db, mean(db))), label="DayaBay constraint")
end

juno_off = [IO_off, NO_off]
for (a, off) in zip(axes, juno_off)
    ylims!(a, 0., 30)
    lines!(a, off[:, 1] .* 1e-3, off[:, 2], label="Juno Δχ²")
end
f[0, 1:2] = Legend(f, [ax_IO, ax_NO], orientation=:horizontal, merge=true)
f
save("combined_output/delta_m31.png", f)
