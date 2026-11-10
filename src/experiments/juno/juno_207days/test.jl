using Distributions
using DensityInterface
using BAT
using DataStructures
using Newtrinos
using LinearAlgebra
using FileIO
using Accessors
using CairoMakie
using DataFrames
using CSV

experiments = (tao=Newtrinos.tao_rewrite.configure(), juno=Newtrinos.juno_rewrite.configure(),)

p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
likelihood = Newtrinos.generate_likelihood(experiments)

# dayabay value as val=sin^2(2 * theta13), and its uncertainty, use chain rule to get error on theta13
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
@reset priors.Δm²₂₁ = Uniform(7.1e-5, 8.0e-5)
@reset priors.θ₁₂ = Uniform(asin(sqrt(0.275)), asin(sqrt(0.32)))
@reset priors.θ₁₃ = Normal(theta13, theta13_sigma)
@reset priors.Δm²₃₁ = Normal(2.541e-3, 0.060e-3)
@reset p.Δm²₃₁ = 2.5413e-3
@reset p.θ₁₃ = theta13
@reset p.Δm²₂₁ = 7.388e-5
@reset p.θ₁₂ = asin(sqrt(0.3036))
if !isdir("combined_output")
    mkdir("combined_output")
end

mle = Newtrinos.find_mle(likelihood, distprod(;priors...), p)
FileIO.save("combined_output/mle.jld2", Dict("mle"=>mle[3]))

f, ax, ax2 = experiments.juno.plot(mle[3])
save("combined_output/juno_spectrum.png", f)
f = experiments.tao.plot(mle[3])
save("combined_output/tao_spectrum.png", f)

