# This example requires Cloudy to be installed.
#using Pkg; Pkg.add(PackageSpec(name="Cloudy", version="0.1.0"))
using Cloudy
const PDistributions = Cloudy.ParticleDistributions

# Import modules
using Distributions  # probability distributions and associated functions
using StatsBase
using LinearAlgebra
using StatsPlots
using Plots
using JLD2
using Random

# Import Calibrate-Emulate-Sample modules
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage

# Import the module that runs Cloudy
include("GModel.jl")
using .GModel

################################################################################
#                                                                              #
#                      Cloudy Calibrate-Emulate-Sample Example                 #
#                                                                              #
#                                                                              #
#     This example uses Cloudy, a microphysics model that simulates the        #
#     coalescence of cloud droplets into bigger drops, to demonstrate how      #
#     the full Calibrate-Emulate-Sample pipeline can be used for Bayesian      #
#     learning and uncertainty quantification of parameters, given some        #
#     observations.                                                            #
#                                                                              #
#     Specifically, this examples shows how to learn parameters of the         #
#     initial cloud droplet mass distribution, given observations of some      #
#     moments of that mass distribution at a later time, after some of the     #
#     droplets have collided and become bigger drops.                          #
#                                                                              #
#     In this example, Cloudy is used in a "perfect model" (aka "known         #
#     truth") setting, which means that the "observations" are generated by    #
#     Cloudy itself, by running it with the true parameter values. In more     #
#     realistic applications, the observations will come from some external    #
#     measurement system.                                                      #
#                                                                              #
#     The purpose is to show how to do parameter learning using                #
#     Calibrate-Emulate-Sample in a simple (and highly artificial) setting.    #
#                                                                              #
#     For more information on Cloudy, see                                      #
#              https://github.com/CliMA/Cloudy.jl.git                          #
#                                                                              #
################################################################################


rng_seed = 41
Random.seed!(rng_seed)


homedir = pwd()
println(homedir)
figure_save_directory = homedir*"/output/"
data_save_directory = homedir*"/output/"
if ~isdir(figure_save_directory)
    mkdir(figure_save_directory)
end
if ~isdir(data_save_directory)
    mkdir(data_save_directory)
end

###
###  Define the (true) parameters and their priors
###

# Define the parameters that we want to learn
# We assume that the true particle mass distribution is a Gamma distribution 
# with parameters N0_true, θ_true, k_true
param_names = ["N0", "u", "k"]
n_param = length(param_names)

N0_true = 300.0  # number of particles (scaling factor for Gamma distribution)
θ_true = 1.5597  # scale parameter of Gamma distribution
k_true = 0.0817  # shape parameter of Gamma distribution
params_true = [N0_true, θ_true, k_true]
# Note that dist_true is a Cloudy distribution, not a Distributions.jl 
# distribution
dist_true = PDistributions.Gamma(N0_true, θ_true, k_true)


###
###  Define priors for the parameters we want to learn
###

# Define constraints
lbound_N0 = 0.4 * N0_true 
lbound_θ = 1.0e-1
lbound_k = 1.0e-4
c1 = bounded_below(lbound_N0)
c2 = bounded_below(lbound_θ)
c3 = bounded_below(lbound_k)
constraints = [[c1], [c2], [c3]]

# We choose to use normal distributions to represent the prior distributions of
# the parameters in the transformed (unconstrained) space. i.e log coordinates
d1 = Parameterized(Normal(4.5, 1.0)) #truth is 5.19
d2 = Parameterized(Normal(0.0, 2.0)) #truth is 0.378
d3 = Parameterized(Normal(-1.0, 1.0))#truth is -2.51
distributions = [d1, d2, d3]


prior_mean = [4.5; 0.0; -1.0]
prior_cov = Array(Diagonal( ones(Float64, length(prior_mean) ) ) )

param_names = ["N0", "u", "k"]

priors = ParameterDistribution(distributions, constraints, param_names)

###
###  Define the data from which we want to learn the parameters
###

data_names = ["M0", "M1", "M2"]
moments = [0.0, 1.0, 2.0]
n_moments = length(moments)


###
###  Model settings
###

# Collision-coalescence kernel to be used in Cloudy
coalescence_coeff = 1/3.14/4/100
kernel_func = x -> coalescence_coeff
kernel = Cloudy.KernelTensors.CoalescenceTensor(kernel_func, 0, 100.0)

# Time period over which to run Cloudy
tspan = (0., 1.0)  


###
###  Generate (artificial) truth samples
###  Note: The observables y are related to the parameters u by:
###        y = G(x1, x2) + η
###

g_settings_true = GModel.GSettings(kernel, dist_true, moments, tspan)
gt = GModel.run_G(params_true, g_settings_true, PDistributions.update_params, 
                  PDistributions.moment, Cloudy.Sources.get_int_coalescence)

n_samples = 100
yt = zeros(length(gt),n_samples)
# In a perfect model setting, the "observational noise" represent the internal
# model variability. Since Cloudy is a purely deterministic model, there is no
# straightforward way of coming up with a covariance structure for this internal
# model variability. We decide to use a diagonal covariance, with entries
# (variances) largely proportional to their corresponding data values, gt.
Γy = convert(Array, Diagonal([100.0, 5.0, 30.0]))
μ = zeros(length(gt))

# Add noise
for i in 1:n_samples
    yt[:, i] = gt .+ rand(MvNormal(μ, Γy))
end

truth = Observations.Obs(yt, Γy, data_names)
truth_sample = truth.mean

###
###  Calibrate: Unscented Kalman Inversion
###


N_iter = 20 # number of iterations
# need to choose regularization factor α ∈ (0,1],  
# when you have enough observation data α=1: no regularization
α_reg =  1.0
# update_freq 1 : approximate posterior covariance matrix with an uninformative prior
#             0 : weighted average between posterior covariance matrix with an uninformative prior and prior
update_freq = 1

process = Unscented(prior_mean, prior_cov, length(truth_sample), α_reg, update_freq)
ukiobj = EnsembleKalmanProcessModule.EnsembleKalmanProcess(truth_sample, truth.obs_noise_cov, process)


                   
# Initialize a ParticleDistribution with dummy parameters. The parameters 
# will then be set in run_G_ensemble
dummy = 1.0
dist_type = PDistributions.Gamma(dummy, dummy, dummy)
g_settings = GModel.GSettings(kernel, dist_type, moments, tspan)



err = zeros(N_iter) 
for i in 1:N_iter

    params_i = mapslices(x -> transform_unconstrained_to_constrained(priors, x),
                         get_u_final(ukiobj); dims=1)

    # # prediction step 
    # u_p = EnsembleKalmanProcessModule.update_ensemble_prediction!(ukiobj) 

    # # define black box parameter to observation map, 
    # # with certain parameter transformation related to imposing some constraints
    # # i.e. θ -> e^θ  -> G(e^θ) = y
    # u_p_trans = mapslices(x -> transform_unconstrained_to_constrained(priors, x), u_p; dims=1)
    g_ens = GModel.run_G_ensemble(params_i, g_settings,
                                  PDistributions.update_params,
                                  PDistributions.moment,
                                  Cloudy.Sources.get_int_coalescence)
    # analysis step 
    EnsembleKalmanProcessModule.update_ensemble!(ukiobj, g_ens) 

    err[i] = get_error(ukiobj)[end] #mean((params_true - mean(params_i,dims=2)).^2)
    println("Iteration: "*string(i)*", Error: "*string(err[i])*" norm(Cov): "*string(norm(ukiobj.process.uu_cov[i])))
end


# UKI results: the mean is in ukiobj.process.u_mean
#              the covariance matrix is in ukiobj.process.uu_cov
transformed_params_true = transform_constrained_to_unconstrained(priors,
                                                                 params_true)

println("True parameters (transformed): ")
println(transformed_params_true)

println("\nUKI results:")
println(get_u_mean_final(ukiobj))


####
θ_mean_arr = hcat(ukiobj.process.u_mean...)
N_θ = length(ukiobj.process.u_mean[1])
θθ_std_arr = zeros(Float64, (N_θ, N_iter+1))
for i = 1:N_iter+1
    for j = 1:N_θ
        θθ_std_arr[j, i] = sqrt(ukiobj.process.uu_cov[i][j,j])
    end
end


ites = Array(LinRange(1, N_iter+1, N_iter+1))
plot(ites,grid=false, θ_mean_arr[1,:], yerror=3.0*θθ_std_arr[1,:],  label="u1")
plot!(ites, fill(transformed_params_true[1], N_iter+1), linestyle=:dash, linecolor=:grey,label=nothing)
plot!(ites,grid=false, θ_mean_arr[2,:], yerror=3.0*θθ_std_arr[2,:], label="u2", xaxis="Iterations")
plot!(ites, fill(transformed_params_true[2], N_iter+1), linestyle=:dash, linecolor=:grey,label=nothing)
plot!(ites,grid=false, θ_mean_arr[3,:], yerror=3.0*θθ_std_arr[3,:], label="u3", xaxis="Iterations")
plot!(ites, fill(transformed_params_true[3], N_iter+1), linestyle=:dash, linecolor=:grey,label=nothing)


#plots
gr(size=(1800,600))

for i in 1:N_iter
    θ_mean, θθ_cov = ukiobj.process.u_mean[i], ukiobj.process.uu_cov[i]
    θ1, θ2, fθ1θ2 = Gaussian_2d(θ_mean[1:2], θθ_cov[1:2,1:2], 100, 100) 
    p1 = contour(θ1, θ2, fθ1θ2)
    plot!(p1,[transformed_params_true[1]], xaxis="u1", yaxis="u2", seriestype="vline",
        linestyle=:dash, linecolor=:red, label = false,
        title = "UKI iteration = " * string(i)
        )
    plot!(p1,[transformed_params_true[2]], seriestype="hline", linestyle=:dash, linecolor=:red, label = "optimum")

    θ2, θ3, fθ2θ3 = Gaussian_2d(θ_mean[2:3], θθ_cov[2:3,2:3], 100, 100) 
    p2 = contour(θ2, θ3, fθ2θ3)
    plot!(p2,[transformed_params_true[2]], xaxis="u2", yaxis="u3", seriestype="vline",
        linestyle=:dash, linecolor=:red, label = false,
        title = "UKI iteration = " * string(i)
        )
    plot!(p2,[transformed_params_true[3]], seriestype="hline", linestyle=:dash, linecolor=:red, label = "optimum")

    θ3, θ1, fθ3θ1 = Gaussian_2d(θ_mean[3:-2:1], θθ_cov[3:-2:1,3:-2:1], 100, 100) 
    p3 = contour(θ3, θ1, fθ3θ1)
    plot!(p3,[transformed_params_true[3]], xaxis="u3", yaxis="u1", seriestype="vline",
        linestyle=:dash, linecolor=:red, label = false,
        title = "UKI iteration = " * string(i)
        )
    plot!(p3,[transformed_params_true[1]], seriestype="hline", linestyle=:dash, linecolor=:red, label = "optimum")

    p=plot(p1, p2, p3, layout=(1,3))    
    display(p)
    sleep(0.5)
end
