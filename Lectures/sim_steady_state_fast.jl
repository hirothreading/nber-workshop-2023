using LinearAlgebra
using Random

"""Part 0: example calibration from notebook"""

function example_calibration()
    y, _, Pi = discretize_income(0.975, 0.7, 7)
    return Dict(
        :a_grid => discretize_assets(0, 10_000, 500),
        :y => y, :Pi => Pi,
        :r => 0.01/4, :beta => 1-0.08/4, :eis => 1
    )
end

"""Part 1: discretization tools"""

function discretize_income(rho, sigma, n_e)
    p = (1 + rho) / 2
    e = 0:(n_e-1)
    alpha = 2 * sigma / sqrt(n_e - 1)
    e = alpha * e
    Pi = rouwenhorst_Pi(n_e, p)
    pi = stationary_markov(Pi)
    y = exp.(e)
    y /= dot(pi, y)
    return y, pi, Pi
end

"""Support for part 1: equality testing and Markov chain convergence"""

function equal_tolerance(x1, x2, tol)
    x1 = vec(x1)
    x2 = vec(x2)
    for i in eachindex(x1)
        if abs(x1[i] - x2[i]) >= tol
            return false
        end
    end
    return true
end

function stationary_markov(Pi, tol=1e-14)
    n = size(Pi, 1)
    pi = fill(1/n, n)
    for it in 1:10_000
        pi_new = Pi' * pi
        if it % 10 == 0 && equal_tolerance(pi, pi_new, tol)
            return pi_new
        end
        pi = pi_new
    end
end

"""Part 2: Backward iteration for policy"""

function backward_iteration(Va, Pi, a_grid, y, r, beta, eis)
    Wa = beta * Pi * Va
    c_endog = Wa .^ (-eis)
    coh = y .+ (1 + r) .* a_grid'
    a = interpolate_monotonic_loop(coh, c_endog .+ a_grid', a_grid)
    setmin!(a, a_grid[1])
    c = coh .- a
    Va = (1 + r) .* c .^ (-1 / eis)
    return Va, a, c
end

function policy_ss(Pi, a_grid, y, r, beta, eis, tol=1e-9)
    coh = y .+ (1 + r) .* a_grid'
    c = 0.05 .* coh
    Va = (1 + r) .* c .^ (-1 / eis)
    for it in 1:10_000
        Va, a, c = backward_iteration(Va, Pi, a_grid, y, r, beta, eis)
        if it % 10 == 1 && equal_tolerance(a, a_old, tol)
            return Va, a, c
        end
        a_old = a
    end
end

"""Support for part 2: equality testing and Markov chain convergence"""

function interpolate_monotonic(x, xp, yp)
    nx, nxp = length(x), length(xp)
    y = similar(x)
    xp_i = 1
    xp_lo = xp[xp_i]
    xp_hi = xp[xp_i + 1]
    for xi_cur in 1:nx
        x_cur = x[xi_cur]
        while xp_i < nxp - 1
            if x_cur < xp_hi
                break
            end
            xp_i += 1
            xp_lo = xp_hi
            xp_hi = xp[xp_i + 1]
        end
        pi = (xp_hi - x_cur) / (xp_hi - xp_lo)
        y[xi_cur] = pi * yp[xp_i] + (1 - pi) * yp[xp_i + 1]
    end
    return y
end

function interpolate_monotonic_loop(x, xp, yp)
    ne = size(x, 1)
    y = similar(x)
    for e in 1:ne
        y[e, :] = interpolate_monotonic(x[e, :], xp, yp)
    end
    return y
end

function setmin!(x, xmin)
    ni, nj = size(x)
    for i in 1:ni
        for j in 1:nj
            if x[i, j] < xmin
                x[i, j] = xmin
            else
                break
            end
        end
    end
end

"""Part 3: forward iteration for distribution"""

function interpolate_lottery(x, xp)
    nx, nxp = length(x), length(xp)
    i = similar(x, Int64)
    pi = similar(x)
    xp_i = 1
    xp_lo = xp[xp_i]
    xp_hi = xp[xp_i + 1]
    for xi_cur in 1:nx
        x_cur = x[xi_cur]
        while xp_i < nxp - 1
            if x_cur < xp_hi
                break
            end
            xp_i += 1
            xp_lo = xp_hi
            xp_hi = xp[xp_i + 1]
        end
        i[xi_cur] = xp_i
        pi[xi_cur] = (xp_hi - x_cur) / (xp_hi - xp_lo)
    end
    return i, pi
end

function interpolate_lottery_loop(x, xp)
    i = similar(x, Int64)
    pi = similar(x)
    for e in 1:size(x, 1)
        i[e, :], pi[e, :] = interpolate_lottery(x[e, :], xp)
    end
    return i, pi
end

function distribution_ss(Pi, a, a_grid, tol=1e-10)
    a_i, a_pi = interpolate_lottery_loop(a, a_grid)
    pi = stationary_markov(Pi)
    D = pi * ones(length(a_grid))' / length(a_grid)
    for it in 1:10_000
        D_new = forward_iteration(D, Pi, a_i, a_pi)
        if it % 10 == 0 && equal_tolerance(D_new, D, tol)
            return D_new
        end
        D = D_new
    end
end

"""Part 4: solving for steady state, including aggregates"""

function steady_state(Pi, a_grid, y, r, beta, eis)
    Va, a, c = policy_ss(Pi, a_grid, y, r, beta, eis)
    D = distribution_ss(Pi, a, a_grid)
    a_i, a_pi = interpolate_lottery_loop(a, a_grid)
    return Dict(
        :D => D, :Va => Va,
        :a => a, :c => c, :a_i => a_i, :a_pi => a_pi,
        :A => dot(a, D), :C => dot(c, D),
        :Pi => Pi, :a_grid => a_grid, :y => y, :r => r, :beta => beta, :eis => eis
    )
end