import numpy as np

def generate_correlated_freqdist(activities, xi, k_out_target, mean_target):
    activities = np.array(activities)
    # Rescale activities:
    activities = activities/np.mean(activities)
    eps = 1e-7
    k_in = (np.mean(activities)/np.std(activities))**2
    N_obs = activities.shape[0]
    if xi < eps:
        f = np.random.gamma(k_in, 1/k_in, N_obs)
    elif xi > 1-eps:
        f = activities
    else:
        f = (1-xi) * np.random.gamma(k_in, 1/k_in, N_obs) + xi * activities

    target_CV = 1/np.sqrt(k_out_target)
    current_CV = np.std(f)/np.mean(f)

    F = f

    while abs(target_CV - current_CV) > 0.01:
        if current_CV > target_CV:
            F -= 0.1 * (F-np.mean(F))
        else:
            F +=  0.1 * (F-np.mean(F))
        current_CV = np.std(F)/np.mean(F)
    k_out = 1/current_CV**2
    F /= np.mean(F)
    F *= mean_target
    # Note: Some values may be returned as 0-|epsilon|, for some small epsilon.
    # To remedy, we shift by the min of the distribution and add epsilon:
    F -= np.min(F)
    F += eps
    F /= np.mean(F)
    F *= mean_target
    return F, np.corrcoef(activities,F)[0,1], k_out
