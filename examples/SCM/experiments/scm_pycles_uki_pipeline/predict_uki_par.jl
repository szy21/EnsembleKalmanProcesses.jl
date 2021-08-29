# This is an example on training the SCAMPy implementation of the EDMF
# scheme with data generated using PyCLES.
#
# The example seeks to find the optimal values of the entrainment and
# detrainment parameters of the EDMF scheme to replicate the LES profiles
# of the BOMEX experiment.
#
# This example is fully parallelized and can be run in the cluster with
# the included script.

# Import modules to all processes
@everywhere using Pkg
@everywhere Pkg.activate("../..")
@everywhere using Distributions
@everywhere using StatsBase
@everywhere using LinearAlgebra
# Import EKP modules
@everywhere using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
@everywhere using EnsembleKalmanProcesses.Observations
@everywhere using EnsembleKalmanProcesses.ParameterDistributionStorage
@everywhere include(joinpath(@__DIR__, "../../src/helper_funcs.jl"))


include(joinpath(@__DIR__, "../../src/ekp_plots.jl"))
using JLD2


""" Define parameters and their priors"""
function construct_priors()
    # Define the parameters that we want to learn
    params = Dict(
        # entrainment parameters
        # "entrainment_factor"                => [bounded(0.0, 1.5)],
        # "detrainment_factor"                => [bounded(0.0, 1.5)],

        "entrainment_factor"                => [bounded(0.0, 2.0*0.13)],
        "detrainment_factor"                => [bounded(0.0, 2.0*0.51)],
        "turbulent_entrainment_factor"      => [bounded(0.0, 2.0*0.015)],
        "entrainment_smin_tke_coeff"        => [bounded(0.0, 2.0*0.3)],
        "updraft_mixing_frac"               => [bounded(0.0, 2.0*0.25)],
        "entrainment_sigma"                 => [bounded(0.0, 2.0*10.0)],
        "sorting_power"                     => [bounded(0.0, 2.0*2.0)],
        "aspect_ratio"                      => [bounded(0.01*0.2, 2.0*0.2)],
    )
    param_names = collect(keys(params))
    constraints = collect(values(params))
    n_param = length(param_names)

    # All vars are standard Gaussians in unconstrained space
    # Initial condition (not necessary the prior)
    σ0 = 0.1
    prior_dist = [Parameterized(Normal(0.0, σ0))
                    for _ in range(1, n_param, length=n_param) ]
    priors = ParameterDistribution(prior_dist, constraints, param_names)
    
    prior_mean = zeros(n_param)
    prior_cov = Array(Diagonal(zeros(n_param) .+ σ0^2))
    
    return priors, prior_mean, prior_cov
end

""" Define reference simulations for loss function"""
function construct_reference_models()::Vector{ReferenceModel}
    les_root = "/groups/esm/zhaoyi/pycles_clima"
    scm_root = pwd()  # path to folder with `Output.<scm_name>.00000` files

    # Calibrate using reference data and options described by the ReferenceModel struct.
    ref_bomex = ReferenceModel(
        # Define variables considered in the loss function
        y_names = ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"],
        # y_names = ["thetal_mean", "qt_mean", "total_flux_h", "total_flux_qt"],
        
        # y_names = ["thetal_mean", "qt_mean"],
        # Reference data specification
        les_root = les_root,
        les_name = "Bomex",
        les_suffix = "aug09",
        # Simulation case specification
        scm_root = scm_root,
        scm_name = "Bomex",
        # Define observation window (s)
        t_start = 4.0 * 3600,  # 4hrs
        t_end = 24.0 * 3600,  # 6hrs
    )

    # Calibrate using reference data and options described by the ReferenceModel struct.
    ref_rico = ReferenceModel(
        # Define variables considered in the loss function
        # y_names = ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"],
        y_names = ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"],
        
        # y_names = ["thetal_mean", "qt_mean"],
        # Reference data specification
        les_root = les_root,
        les_name = "Rico",
        les_suffix = "aug23",
        # Simulation case specification
        scm_root = scm_root,
        scm_name = "Rico",
        # Define observation window (s)
        t_start = 4.0 * 3600,  # 4hrs
        t_end = 24.0 * 3600,  # 6hrs
    )

    # Make vector of reference models
    ref_models::Vector{ReferenceModel} = [ref_bomex, ref_rico]
    @assert all(isdir.([les_dir.(ref_models)... scm_dir.(ref_models)...]))

    return ref_models
end

function run_predict()
    #########
    #########  Define the parameters and their priors
    #########
    priors, prior_mean, prior_cov = construct_priors()
    

    #########
    #########  Define simulation parameters and data directories
    #########
    ref_models = construct_reference_models()

    outdir_root = pwd()
    # Define preconditioning and regularization of inverse problem
    perform_PCA = false # Performs PCA on data
    # todo
    normalize = true  # whether to normalize data by pooled variance
    # Flag to indicate whether reference data is from a perfect model (i.e. SCM instead of LES)
    model_type::Symbol = :les  # :les or :scm
    
    #########
    #########  Retrieve true LES samples from PyCLES data and transform
    #########

    
    # Compute data covariance
    ref_stats = ReferenceStatistics(ref_models, model_type, perform_PCA, normalize)
    d = length(ref_stats.y) # Length of data array

    #########
    #########  Calibrate: Ensemble Kalman Inversion
    #########
    # need to choose regularization factor α ∈ (0,1],  
    # when you have enough observation data α=1: no regularization
    α_reg =  1.0
    # update_freq 1 : approximate posterior covariance matrix with an uninformative
    #                 prior
    #             0 : weighted average between posterior covariance matrix with an
    #                 uninformative prior and prior
    update_freq = 1

    # prior distribution : prior_cov = nothing (uninformative prior)
    algo   = Unscented(prior_mean, prior_cov, d, α_reg, update_freq; prior_cov = nothing) # 100*prior_cov)  
    N_ens  = 2*length(prior_mean) + 1 # number of ensemble members
    N_iter = 20 # number of EKP iterations.

    # parameters are sampled in unconstrained space
    ekobj = EnsembleKalmanProcess(ref_stats.y, ref_stats.Γ, algo )

    # Define caller function
    @everywhere g_(x::Vector{FT}) where FT<:Real = run_SCM(
        x, $priors.names, $ref_models, $ref_stats,
    )

    # Create output dir
    algo_type = "uki"
    n_param = length(priors.names)
    ekp_path = joinpath(outdir_root, "results_$(algo_type)_p$(n_param)_e$(N_ens)_i$(N_iter)_d300_$(model_type)")
    println("Name of outdir path for this EKP is: $ekp_path")
    mkpath(ekp_path)

    # EKP iterations
    g_ens = zeros(N_ens, d)
    
    data = load(joinpath(ekp_path, "ekp.jld2"))

    ekp_mean = data["ekp_mean"]
    ekp_cov = data["ekp_cov"]
    truth_mean = data["truth_mean"]
    inflation_factor = length(truth_mean)/length(ekp_mean)

    @info "inflation factor is : ", inflation_factor
    # x_mean x_cov
    u_final = construct_sigma_ensemble(algo, ekp_mean[end], ekp_cov[end]*inflation_factor) 


    
    # Parameters are transformed to constrained space when used as input to TurbulenceConvection.jl
    params_cons = transform_unconstrained_to_constrained(priors, u_final)
    params = [c[:] for c in eachcol(params_cons)]
    @everywhere params = $params

    array_of_tuples = pmap(g_, params) # Outer dim is params iterator

    (sim_dirs_arr, g_ens_arr, g_ens_arr_pca) = ntuple(l->getindex.(array_of_tuples,l),3) # Outer dim is G̃, G 
    for j in 1:N_ens
        if perform_PCA
            g_ens[j, :] = g_ens_arr_pca[j]
        else
            g_ens[j, :] = g_ens_arr[j]
        end
    end

    # todo only support 1 case
    # Observation plot
    
    truth_mean = ekobj.obs_mean  
    pred_obs = mean(g_ens', dims=2)[:]
    pred_obs_std = sqrt.(diag( cov(g_ens', dims=2) ) )[:]

    pool_var = ref_stats.norm_vec
    n_vars = length(pool_var[1])

    # plot parameter evolution
    n_models = length(ref_models)
    @info "Number of models : ", n_model
    fig, axs = subplots(ncols=n_vars, nrows=n_models, sharey=true, figsize=(4*n_vars, 15*n_models))

    y_ind = 1
    for j = 1:n_models
        z_scm = get_profile(scm_dir(ref_models[j]), ["z_half"])
        y_names = ref_models[j].y_names
        for i = 1:n_vars
            ax = (n_models == 1 ? axs[i] : axs[j,i])
            ax.plot(truth_mean[y_ind:y_ind+length(z_scm)-1], z_scm,       label="Ref")
            ax.plot(pred_obs[y_ind:y_ind+length(z_scm)-1],   z_scm, "--", label="Sim")

            ax.fill_betweenx(z_scm,  
                pred_obs[y_ind:y_ind+length(z_scm)-1] .- 2pred_obs_std[y_ind:y_ind+length(z_scm)-1], 
                pred_obs[y_ind:y_ind+length(z_scm)-1] .+ 2pred_obs_std[y_ind:y_ind+length(z_scm)-1], 
                alpha=0.5,
            )

            ax.set_xlabel(y_names[i])
            ax.legend()
            y_ind += length(z_scm)
        end
    end

    savefig(joinpath(ekp_path, "observations-Pred.png"))

end


