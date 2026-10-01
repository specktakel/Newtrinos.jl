using Distributions
using DensityInterface
using BAT
using DataStructures
using Newtrinos
using FileIO
using Accessors
using CairoMakie
using CSV, DataFrames

experiments = (dayabay = Newtrinos.dayabay_rewrite.configure(),)

likelihood = Newtrinos.generate_likelihood(experiments);

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

#@reset priors.Δm²₃₁ = Uniform(0.0023, 0.0028)
#@reset priors.θ₁₃ = Uniform(0.13, 0.16)

@reset priors.θ₁₂ = p.θ₁₂
@reset priors.θ₂₃ = p.θ₂₃
@reset priors.δCP = p.δCP
@reset priors.Δm²₂₁ = p.Δm²₂₁

result = Newtrinos.find_mle(likelihood, distprod(;priors...), p)

FileIO.save("test_output/test_rewrite.jld2", Dict("result" => result))

experiments.dayabay.plot(result[3])

