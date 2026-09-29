using LinearAlgebra
using HDF5
using Test
using Newtrinos

@testset "DayaBay" begin
    db = Newtrinos.dayabay_rewrite.configure()
    @testset "IAV" begin
        iav_func = Newtrinos.dayabay_rewrite.get_iav_matrix()
        iav = iav_func(1.0)
        N = size(iav)[1]
        for i in 1:20
            delta = zeros(N)
            delta[i] = 1.0
            @test isapprox(sum(iav[:, i]), 0.0)
            @test isapprox(sum(iav * delta), 0.0)
            break
        end
        for i in 21:N
            delta = zeros(N)
            delta[i] = 1.0
            @test isapprox(sum(iav[:, i]), 1.0)
            @test isapprox(sum(iav * delta), 1.0)
            break
        end
        scaled = iav_func(2)
        @test isapprox(scaled[1:99, 100], 2.0 .* iav[1:99, 100])
        @test isapprox(scaled[101:end, 100], 2.0 .* iav[101:end, 100])
        @test isapprox(Diagonal(iav_func(1.0)), Diagonal(iav_func(2.0)))
    end

    @testset "LSNL" begin
        lsnl_data = Newtrinos.dayabay_rewrite.read_lsnl_correction()
        lsnl = Newtrinos.dayabay_rewrite.get_lsnl_correction()
        @test isapprox(lsnl(lsnl_data.E, [0, 0, 0, 0]), lsnl_data.f_nom)
        @test isapprox(lsnl(lsnl_data.E, [1, 0, 0, 0]), lsnl_data.f_nom .+ lsnl_data.rel_0)
        @test isapprox(lsnl(lsnl_data.E, [0, 1, 0, 0]), lsnl_data.f_nom .+ lsnl_data.rel_1)
        @test isapprox(lsnl(lsnl_data.E, [0, 0, 1, 0]), lsnl_data.f_nom .+ lsnl_data.rel_2)
        @test isapprox(lsnl(lsnl_data.E, [0, 0, 0, 1]), lsnl_data.f_nom .+ lsnl_data.rel_3)

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
        @test length(params.iav_offdiag_scale) == length(assets.df_exp[!, "AD"])
        @test length(params.lsnl_pull) == 4
    end
end