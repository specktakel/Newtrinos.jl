module ibd_xsec

using DelimitedFiles
using Interpolations
using PCHIPInterpolation
using DataFrames
using CSV
using ..Newtrinos



"""
    Abstract type for inverse beta decay cross-section models.
"""
abstract type IBDModel end


"""
    StrumiaVissani <: IBDModel

Tabulated cross-section of Strumia and Vissani (https://arxiv.org/abs/astro-ph/0302055).
"""
struct StrumiaVissani <: IBDModel end

@kwdef struct IBDXsec <: Newtrinos.Physics
    cfg::IBDModel
    params::NamedTuple
    priors::NamedTuple
    xsec::Function
    Epos::Function
    Enu::Function
end


function configure(cfg::IBDModel=StrumiaVissani())
    IBDXsec(
        cfg = cfg,
        params = get_params(cfg),
        priors = get_priors(cfg),
        xsec = get_xsec(cfg)[1],
        Epos = get_xsec(cfg)[2],
        Enu = get_xsec(cfg)[3],
    )
end


function get_params(cfg::StrumiaVissani)
    return (;)
end


function get_priors(cfg::StrumiaVissani)
    return (;)
end


function get_xsec(cfg::StrumiaVissani)
    # create spline of cross section from table
    xsec = CSV.read(joinpath(@__DIR__, "ibd_xsec.csv"), DataFrame, header=0, delim=",")
    energies = reshape(Matrix(xsec[!, [1, 5, 9]]), (45))
    xsection = reshape(Matrix(xsec[!, [2, 6, 10]]), (45))
    #Epos1 = [parse(Float64, i) for i in collect(xsec[!, 3])[2:end]]
    #Epos2 = reshape(Matrix(xsec[!, [7, 11]]), (30))
    #println(Epos1)
    #println(Epos2)
    #Epos = vcat(Epos1, Epos2)
    Epos = reshape(Matrix(xsec[!, [3, 7, 11]]), (45))
    _Epos_itp = Interpolator(energies, Epos, extrapolate=true);
    _x_sec_itp = Interpolator(energies, xsection, extrapolate=true);
    _Enu_itp = Interpolator(Epos, energies, extrapolate=true);

    # impose zero as lower bound
    function x_sec_itp(x)
        out = _x_sec_itp(x)
        if out < 0
            return 0.
        else return out
        end
    end

    function E_pos_interp(x)
        out = _Epos_itp(x)
        #if x < energies[2]
        if x < energies[1]
            return 0.
        else
            return out
        end
    end

    function E_nu_interp(x)
        # maps from deposited to neutrino,
        # hence subtract electron mass
        out = _Enu_itp.(x) - 0.511
    end


    return x_sec_itp, E_pos_interp, E_nu_interp
end

end