module ibd_xsec

using DelimitedFiles
using Interpolations
using DataStructures
using ..Newtrinos



"""
    Abstract type for inverse beta decay cross-section models.
"""
abstract type IBDModel <: Newtrinos.Physics.xsec.XsecModel end


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
end


function configure(cfg::IBDModel=StrumiaVissani())
    IBDXsec(
        cfg = cfg,
        params = get_params(cfg),
        priors = get_priors(cfg),
        xsec = get_xsec(cfg)
    )
end


function get_params(cfg::StrumiaVissani)
    return (,)
end


function get_priors(cfg::StrumiaVissani)
    return (,)
end


function get_xsec(cfg::StrumiaVissani)
    # create spline of cross section from table
    xsec = CSV.read(joinpath(@__DIR__, "ibd_xsec.csv"), DataFrame, header=0, delim=",")
    energies = reshape(Matrix(xsec[!, [1, 5, 9]]), (45))
    xsection = reshape(Matrix(xsec[!, [2, 6, 10]]), (45))
    _x_sec_itp = Interpolator(energies, xsection, extrapolate=true);

    # impose zero as lower bound
    function x_sec_itp(x)
        out = _x_sec_itp(x)
        if out < 0
            return 0.
        else return out
        end
    end

    return x_sec_itp
end

end