using JLD2
using PyPlot
using Statistics

function make_ekp_plots(ekp_path::ST, param_names::Vector{ST}; ref_params = nothing) where ST<:AbstractString
    n_param = length(param_names)

    # Load data
    data = load(joinpath(ekp_path, "ekp.jld2"))

    # Mean
    phi_m = mean(data["phi_params"], dims=3)[:,:,1]

    # Variance
    _ustd = std.(data["ekp_u"], dims=2)
    n_iter = length(_ustd); n_param = length(_ustd[1])
    ustd = zeros((n_iter, n_param))
    for i in 1:n_iter ustd[i,:] = _ustd[i] end

    # plot parameter evolution
    fig, axs = subplots(nrows=n_param, sharex=true, figsize=(15, 4*n_param))
    x = 0:n_iter-1
    for (i, ax) in enumerate(axs)
        ax.plot(x, phi_m[:,i])
        if ref_params !== nothing
            ax.plot(x, ones(length(x)) * ref_params[param_names[i]], "--")
        end
        ax.fill_between(x, 
            phi_m[:,i].-2ustd[:,i], 
            phi_m[:,i].+2ustd[:,i], 
            alpha=0.5,
        )
        ax.set_ylabel(param_names[i])
    end

    axs[1].set_xlim(0,n_iter-1)
    axs[1].set_title("Parameter evolution (mean ±2 std)")
    axs[end].set_xlabel("iteration")
    savefig(joinpath(ekp_path, "param_evol.png"))

    # Error plot
    x = 1:n_iter-1
    err = data["ekp_err"]
    fig, ax = subplots()
    ax.plot(x, err)
    ax.set_xlim(1,n_iter-1)
    ax.set_ylabel("Error")
    ax.set_xlabel("iteration")
    ax.set_title("Error evolution")
    savefig(joinpath(ekp_path, "error_evol.png"))

end




function make_ekp_obs_plot(ekp_path::ST, param_names::Vector{ST}, ref_models, iter=nothing) where ST<:AbstractString

    # Load data
    data = load(joinpath(ekp_path, "ekp.jld2"))

    # todo only support 1 case
    # Observation plot
    pool_var = data["pool_var"]
    truth_mean = data["truth_mean"]
    pred_obs = mean(data["ekp_g"][end], dims=2)[:]
    n_vars = length(pool_var[1])

    # plot parameter evolution
    n_models = length(ref_models)
    fig, axs = subplots(ncols=n_vars, nrows=n_models, sharey=true, figsize=(4*n_vars, 15*n_models))

    y_ind = 1
    for j = 1:n_models
        z_scm = get_profile(scm_dir(ref_models[j]), ["z_half"])
        y_names = ref_models[j].y_names
        for i = 1:n_vars
            ax = (n_models == 1 ? axs[i] : axs[j,i])
            ax.plot(truth_mean[y_ind:y_ind+length(z_scm)-1], z_scm,       label="Ref")
            ax.plot(pred_obs[y_ind:y_ind+length(z_scm)-1],   z_scm, "--", label="Sim" )
            ax.set_xlabel(y_names[i])
            ax.legend()
            y_ind += length(z_scm)

           
        end
    end
    savefig(joinpath(ekp_path, (isnothing(iter) ? "observations.png" : "observations-$(iter).png") ))

end
