"""
Simple code for standard incomplete markets model.
Built up in "Lecture 1, Standard Incomplete Markets Steady State.ipynb".
"""

using LinearAlgebra
using Random
using Optim
using Plots
using LaTeXStrings
using Printf
using Interpolations

"""Part 0: example calibration from notebook"""

function example_calibration()
    y, _, Pi = discretize_income(0.975, 0.7, 7)
    return Dict(
        :a_grid => discretize_assets(0, 10000, 500),
        :y => y,
        :Pi => Pi,
        :r => 0.01 / 4,
        :beta => 1 - 0.08 / 4,
        :eis => 1
    )
end

"""Part 1: discretisation tools"""

function discretize_assets(amin, amax, n_a)
    # find maximum ubar of uniform grid corresponding to desired maximum amax of asset grid
    ubar = log(1 + log(1 + amax - amin))
    
    # make uniform grid
    u_grid = range(0, stop=ubar, length=n_a)
    
    # double-exponentiate uniform grid and add amin to get grid from amin to amax
    return amin .+ exp.(exp.(u_grid) .- 1) .- 1
end

function rouwenhorst_Pi(N, p)
    # base case Pi_2
    Pi = [p 1 - p; 1 - p p]
    
    # recursion to build up from Pi_2 to Pi_N
    for n in 3:N
        Pi_old = Pi
        Pi = zeros(n, n)
        
        Pi[1:end-1, 1:end-1] += p * Pi_old
        Pi[1:end-1, 2:end] += (1 - p) * Pi_old
        Pi[2:end, 1:end-1] += (1 - p) * Pi_old
        Pi[2:end, 2:end] += p * Pi_old
    end
    
    return Pi
end

function stationary_markov(Pi, tol=1e-14)
    # start with uniform distribution over all states
    n = size(Pi, 1)
    pi = fill(1/n, n)
    
    # update distribution using Pi until successive iterations differ by less than tol
    for _ in 1:10_000
        pi_new = Pi' * pi #transpose of Pi
        if maximum(abs.(pi_new - pi)) < tol
            return pi_new
        end
        pi = pi_new
    end
end

function discretize_income(rho, sigma, n_e)
    # choose inner-switching probability p to match persistence rho
    p = (1 + rho) / 2
    
    # start with states from 0 to n_e-1, scale by alpha to match standard deviation sigma
    e = collect(0:n_e-1)
    alpha = 2 * sigma / sqrt(n_e - 1)
    e = alpha .* e
    
    # obtain Markov transition matrix Pi and its stationary distribution
    Pi = rouwenhorst_Pi(n_e, p)
    pi = stationary_markov(Pi)
    
    # e is log income, get income y and scale so that mean is 1
    y = exp.(e)
    y /= dot(pi, y)
    
    return y, pi, Pi
end


"""Part 2: Backward iteration for policy"""

function backward_iteration(Va, Pi, a_grid, y, r, beta, eis)
    # step 1: discounting and expectations
    Wa = (beta * Pi) * Va
    
    # step 2: solving for asset policy using the first-order condition
    c_endog = Wa.^(-eis)
    coh = y .+ (1+r) .* a_grid'
    
    a = similar(coh)
    for e in 1:length(y)
        itp = interpolate((c_endog[e, :] .+ a_grid,), a_grid, Gridded(Linear()))
        extrap_itp = extrapolate(itp, Interpolations.Flat())
        a[e, :] = extrap_itp(coh[e, :])
    end
    
    # step 3: enforcing the borrowing constraint and backing out consumption
    a = max.(a, a_grid[1])
    c = coh .- a
    
    # step 4: using the envelope condition to recover the derivative of the value function
    Va = (1+r) .* c.^(-1/eis)
    
    return Va, a, c
end

function policy_ss(Pi, a_grid, y, r, beta, eis, tol=1e-9)
    # initial guess for Va: assume consumption 5% of cash-on-hand, then get Va from envelope condition
    coh = y .+ (1+r) .* a_grid'
    c = 0.05 .* coh
    Va = (1+r) .* c.^(-1/eis)
    
    # Initialize a_old with the same shape as coh
    a_old = similar(coh)
    
    # iterate until maximum distance between two iterations falls below tol, fail-safe max of 10,000 iterations
    for it in 1:10_000
        Va, a, c = backward_iteration(Va, Pi, a_grid, y, r, beta, eis)
        
        # after iteration 0, can compare new policy function to old one
        if it > 1 && maximum(abs.(a - a_old)) < tol
            return Va, a, c
        end
        
        a_old .= a  # Update a_old with the new a
    end
end


"""Part 3: forward iteration for distribution"""
function get_lottery(a, a_grid)
    if isa(a, AbstractArray)
        # Initialize arrays to store the results
        a_i = similar(a, Int)
        a_pi = similar(a)
        
        # Iterate over each element in `a`
        for i in eachindex(a)
            idx = searchsortedfirst(a_grid, a[i])
            if idx == 1
                a_i[i] = 1
                a_pi[i] = 1.0
            elseif idx > length(a_grid)
                a_i[i] = length(a_grid) - 1
                a_pi[i] = 0.0
            else
                a_i[i] = idx - 1
                a_pi[i] = (a_grid[idx] - a[i]) / (a_grid[idx] - a_grid[idx-1])
            end
        end
    else
        # Handle scalar input
        idx = searchsortedfirst(a_grid, a)
        if idx == 1
            a_i = 1
            a_pi = 1.0
        elseif idx > length(a_grid)
            a_i = length(a_grid) - 1
            a_pi = 0.0
        else
            a_i = idx - 1
            a_pi = (a_grid[idx] - a) / (a_grid[idx] - a_grid[idx-1])
        end
    end
    
    return a_i, a_pi
end

function forward_policy(D, a_i, a_pi)
    Dend = zeros(size(D))
    @inbounds for e in 1:size(a_i, 1)
        @simd for a in 1:size(a_i, 2) # SIMD macro is added to the inner loop to enable SIMD optmisations
            # send pi(e,a) of the mass to gridpoint i(e,a)
            Dend[e, a_i[e,a]] += a_pi[e,a] * D[e,a]
            
            # send 1-pi(e,a) of the mass to gridpoint i(e,a)+1
            Dend[e, a_i[e,a]+1] += (1 - a_pi[e,a]) * D[e,a]
        end
    end
    return Dend
end

function forward_iteration(D, Pi, a_i, a_pi)
    Dend = forward_policy(D, a_i, a_pi)
    return Pi' * Dend # transpose of Pi
end

function distribution_ss(Pi, a, a_grid, tol=1e-10)
    a_i, a_pi = get_lottery(a, a_grid) # get_lottery() is called to compute a_i and a_pi
    
    # as initial D, use stationary distribution for e, plus uniform over a
    pi = stationary_markov(Pi) # stationary_markov() is called to compute stationary distribution of pi
    D = pi * ones(length(a_grid))' / length(a_grid) #initialise D with the stationary distribution for e and a uniform distribution over a
    
    # now iterate until convergence to acceptable threshold
    # loop iterates until maximum distance between two iterations falls below tol, fail-safe max of 10,000 iterations
    for _ in 1:10_000
        D_new = forward_iteration(D, Pi, a_i, a_pi)
        if maximum(abs.(D_new .- D)) < tol
            return D_new
        end
        D = D_new
    end
    return D  # Return D if the loop completes without convergence
end


"""Part 4: solving for steady state, including aggregates"""
function steady_state(Pi, a_grid, y, r, beta, eis)
    Va, a, c = policy_ss(Pi, a_grid, y, r, beta, eis)
    a_i, a_pi = get_lottery(a, a_grid)
    D = distribution_ss(Pi, a, a_grid)
    
    # Aggregation
    A = dot(a, D) # Aggregation. Equivalent to np.vdot(a, D) in Python 
    C = dot(c, D) # Aggregation. Equivalent to np.vdot(c, D) in Python
    
    return Dict(
        :D => D,
        :Va => Va,
        :a => a,
        :a_i => a_i,
        :a_pi => a_pi,
        :c => c,
        :A => A,
        :C => C,
        :Pi => Pi,
        :a_grid => a_grid,
        :y => y,
        :r => r,
        :beta => beta,
        :eis => eis
    )
end


"""Part 5: Expectation iterations"""
function expectation_policy(Xend, a_i, a_pi)
    X = zeros(size(Xend))
    @inbounds for e in 1:size(a_i, 1)
        @simd for a in 1:size(a_i, 2)
            # expectation is pi(e,a)*Xend(e,i(e,a)) + (1-pi(e,a))*Xend(e,i(e,a)+1)
            X[e, a] = a_pi[e, a]*Xend[e, a_i[e, a]] + (1-a_pi[e, a])*Xend[e, a_i[e, a]+1]
        end
    end
    return X
end

function expectation_iteration(X, Pi, a_i, a_pi)
    Xend = Pi * X
    return expectation_policy(Xend, a_i, a_pi)
end

function expectation_functions(X, Pi, a_i, a_pi, T)
    # set up array of curlyE and fill in first row with base case
    curlyE = Array{eltype(X)}(undef, T, size(X, 1), size(X, 2))
    curlyE[1, :, :] .= X
    
    # recursively apply law of iterated expectations
    for j in 2:T
        curlyE[j, :, :] .= expectation_iteration(curlyE[j-1, :, :], Pi, a_i, a_pi)
    end
    
    return curlyE
end