using Test
using Newtrinos

@testset "DayaBay" begin
    db = Newtrinos.dayabay_rewrite.configure()
    @testset "IAV" begin
        iav = Newtrinos.dayabay_rewrite.get_iav_matrix()
        N = size(iav)[1]
        for i in 1:20
            delta = zeros(N)
            delta[i] = 1.0
            @test isapprox(sum(iav[:, i]), 0.0)
            @test isapprox(sum(iav * delta), 0.0)
        end
        for i in 21:N
            delta = zeros(N)
            delta[i] = 1.0
            @test isapprox(sum(iav[:, i]), 1.0)
            @test isapprox(sum(iav * delta), 1.0)
        end

        # TODO: offdiagonal
    end

    @testset "LSNL" begin
        lsnl_data = Newtrinos.dayabay_rewrite.read_lsnl_correction()
        lsnl_interp = Newtrinos.dayabay_rewrite.get_lsnl_correction()
        @test isapprox(lsnl_interp.(lsnl_data.E), lsnl_data.f_nom)

        #TODO pull curves
    end

    @testset "ERES" begin
        params = db.params
        E = collect(LinRange(1, 12, 200))
        sigma_E = Newtrinos.dayabay_rewrite.eres(E, params.eres_a, params.eres_b, params.eres_c)
        for i in 30:length(E) - 30
            flux = zeros(length(E))
            flux[i] = 1
            @test isapprox(sum(Newtrinos.dayabay_rewrite.smear(E, flux, sigma_E, width=10)), 1., rtol=1e-3)
        end
    end

    assets = db.assets
    params = Newtrinos.get_params((;db))
    @testset "assets" begin
        fwd = db.forward_model
        @test length(db.assets.observed) == length(fwd(params))
    end

    @testset "parameters" begin
        @test length(params.acc_scale) == length(assets.detectors_6AD) + length(assets.detectors_8AD) + length(assets.detectors_7AD)
        @test length(params.alpha_n_rate) == length(assets.detectors_6AD) + length(assets.detectors_8AD) + length(assets.detectors_7AD)
        @test length(params.fast_n_unc_scale) == length(assets.EH_list) * length(assets.period_list)
        @test length(params.lihe_unc_scale) == length(assets.EH_list) * length(assets.period_list)
    end
end