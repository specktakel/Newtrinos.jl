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
        val1 = flux(3,  params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, 0, 0, params.spectrum_pulls, 1)
        val2 = flux(2,  params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, 0, 0, params.spectrum_pulls, 1)
        vals1_2 = flux([3, 2],  params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, 0, 0, params.spectrum_pulls, 1)
        @test isapprox(vals1_2, [val1, val2])
        val1 = flux(3,  params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, 0, 1, params.spectrum_pulls, 1)
        val2 = flux(3,  params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, 0, 0, params.spectrum_pulls, 1)
        @test val1 / val2 <= 1.04
        
        @reset params.reactor_thermal_power_scale = 1.0
        flux_vals = flux(E, params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, 0, 0, params.spectrum_pulls, 1)
        @reset params.reactor_thermal_power_scale = 2.0
        flux_val_higher = flux(E, params.reactor_thermal_power_scale[1], params.energy_per_fission, params.fission_fractions_scale_R1, 0, 0, params.spectrum_pulls, 1)
        @test isapprox(flux_vals .* 2, flux_val_higher)
    end

end