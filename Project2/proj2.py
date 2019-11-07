import numpy as np
import pandas as pd
import datetime
from scipy.io import loadmat
from scipy.special import digamma, logsumexp
import json
import timeit


def lda_inference(w, beta, alpha=0.1):
    K = beta.shape[1]
    phi = np.full((len(w), K), 1/K)
    gamma = np.full(K, alpha + len(w) / K)
    iterations = 0
    while True:
        iterations += 1
        prev_phi = phi
        prev_gamma = gamma

        phi = np.log(beta[w, :]) + digamma(gamma)
        phi = np.exp(phi)
        phi /= np.sum(phi, axis=1, keepdims=True)
        gamma = alpha + phi.sum(axis=0)

        if np.all(np.abs(phi - prev_phi) < 1e-3) and np.all(np.abs(gamma - prev_gamma) < 1e-3):
            break

    return (phi, gamma, iterations)


def infer_all(D, beta, alpha):
    for d in D:
        w = np.repeat(np.arange(len(d)), d)
        phi, gamma, iterations = lda_inference(w, beta, alpha)
        gamma /= gamma.sum()


if __name__ == "__main__":
    # Load data
    data = loadmat("proj2_data.mat")
    D = data["data"]
    beta = data["beta_matrix"]

    # Infer first individual and save phi
    w = np.repeat(np.arange(len(D[0])), D[0])
    phi, gamma, iterations = lda_inference(w, beta)
    with open('phi1.out', 'w') as f:
        json.dump(phi.tolist(), f)

    # Infer all individuals and save Theta
    Theta = []
    for d in D:
        w = np.repeat(np.arange(len(d)), d)
        phi, gamma, iterations = lda_inference(w, beta)
        gamma /= gamma.sum()
        Theta.append(gamma)
    Theta = np.array(Theta)
    with open('Theta.out', 'w') as f:
        json.dump(Theta.tolist(), f)

    # Perform inference on all individuals
    # with varying alpha
    stats = []
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        for d in D:
            w = np.repeat(np.arange(len(d)), d)
            phi, gamma, iterations = lda_inference(w, beta, alpha)
            gamma /= gamma.sum()
            stats.append({
                "alpha": alpha,
                "g0": gamma[0],
                "g1": gamma[1],
                "g2": gamma[2],
                "g3": gamma[3],
                "iterations": iterations,
                "genotype length": np.sum(d)
            })
    df = pd.DataFrame(stats)
    df.to_csv("varying_alpha.csv")

    # Time the performance of inference on varying alpha
    timer_stats = []
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        time = timeit.timeit('infer_all(D, beta, alpha)', globals=locals(), number=100)
        timer_stats.append(
            {
                "alpha": alpha,
                "time": time
            })
    df = pd.DataFrame(timer_stats)
    df.to_csv("timer_stats.csv")
