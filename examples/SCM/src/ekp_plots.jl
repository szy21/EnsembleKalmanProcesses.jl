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

    # Plot parameter evolution
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



function make_ekp_plots(ekp_path::ST, param_names::Vector{ST}, uki; ref_params = nothing) where ST<:AbstractString
    n_param = length(param_names)


    # # Load data
    # data = load(joinpath(ekp_path, "ekp.jld2"))

    # data["ekp_u"]
    # # Mean
    # phi_m = mean(data["phi_params"], dims=3)[:,:,1]

    # # Variance
    # _ustd = std.(data["ekp_u"], dims=2)
    # n_iter = length(_ustd); n_param = length(_ustd[1])
    # ustd = zeros((n_iter, n_param))
    # for i in 1:n_iter ustd[i,:] = _ustd[i] end

    # Load data
    data = load(joinpath(ekp_path, "ekp.jld2"))

    # # Mean
    # phi_m = mean(data["phi_params"], dims=3)[:,:,1]
    # # Variance
    # _ustd = std.(data["ekp_u"], dims=2)

    ekp_u = data["ekp_u"]
    n_iter = length(ekp_u)

    @info "size(ekp_u[1]) = ", size(ekp_u[1])
    n_param = size(ekp_u[1], 1)

    ustd = zeros((n_iter, n_param))
    phi_m = zeros((n_iter, n_param))

    for i = 1:n_iter
        uki_mean = construct_mean(uki, ekp_u[i])  
        uki_cov = construct_cov(uki, ekp_u[i], uki_mean)    

        phi_m[i, :] = uki_mean
        ustd[i, :] = sqrt.(diag(uki_cov))
    end

    # Plot parameter evolution
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


function make_ekp_obs_plot(ekp_path::ST, param_names::Vector{ST}, ref_models, uki, iter=nothing) where ST<:AbstractString

    # Load data
    data = load(joinpath(ekp_path, "ekp.jld2"))

    # todo only support 1 case
    # Observation plot
    pool_var = data["pool_var"]
    truth_mean = data["truth_mean"]

    # pred_obs = mean(data["ekp_g"][end], dims=2)[:]
    # pred_obs_std = sqrt.(diag( cov(data["ekp_g"][end], dims=2) ) )[:]

    g_ens = data["ekp_g"][end]
    pred_obs = construct_mean(uki, g_ens)  #mean(g_ens', dims=2)[:]
    pred_cov = construct_cov(uki, g_ens, pred_obs)
    pred_obs_std = sqrt.(diag( pred_cov ) )[:]


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

    @info "Iteration $(iter), empirical model error covariance is : ", mean((pred_obs - truth_mean).^2)
    savefig(joinpath(ekp_path, (isnothing(iter) ? "observations.png" : "observations-$(iter).png") ))

end
