using Test
using Accessors
using Newtrinos

@testset "Reactor flux" begin 
    @testset "Configuration" begin
        reactor = Newtrinos.reactor_flux.configure()
        @test reactor isa Newtrinos.reactor_flux.ReactorFlux
        # TODO: how to comapre that each key of params is in priors?
        params = Newtrinos.reactor_flux.get_params(reactor.cfg.flux_model)
        priors = Newtrinos.reactor_flux.get_priors(reactor.cfg.flux_model)
    end

    @testset "Flux values" begin
        E = collect(LinRange(1, 12, 100))
        reactor = Newtrinos.reactor_flux.configure()
        flux = reactor.flux
        params = reactor.params
        @reset params.reactor_thermal_power_scale = 1.0
        flux_vals = flux(E, params.reactor_thermal_power_scale, params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls)
        @reset params.reactor_thermal_power_scale = 2.0
        flux_val_higher = flux(E, params.reactor_thermal_power_scale, params.energy_per_fission, params.fission_fractions_scale_R1, params.spectrum_pulls)
        @test isapprox(flux_vals .* 2, flux_val_higher)
    end
end