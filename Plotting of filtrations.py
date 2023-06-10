import numpy as np
from ripser import ripser
from persim import plot_diagrams, persistent_entropy, persistent_persistence_image
import matplotlib.pyplot as plt

def calculate_p0(x):
    # Calculate P0(x) via sublevel set filtration
    p0_diagram = ripser(x, maxdim=0)['dgms'][0]
    p0_feature = persistent_entropy(p0_diagram)

    # Plot persistence diagram
    plot_diagrams(p0_diagram, show=True)

    return p0_feature

def calculate_pi(x, dim, lag):
    # Calculate Pi(R_{dim, lag}(x)) via Vietoris-Rips complex filtration
    embedded_x = takens_embedding(x, dim, lag)
    pi_diagram = ripser(embedded_x)['dgms'][dim]
    pi_feature = persistent_entropy(pi_diagram)

    # Plot persistence diagram
    plot_diagrams(pi_diagram, show=True)

    return pi_feature

def calculate_persistence_statistics(x):
    # Calculate Persistence statistics features
    p0_feature = calculate_p0(x)
    pi_feature_120_1 = calculate_pi(x, dim=120, lag=1)
    pi_feature_1_1 = calculate_pi(x, dim=1, lag=1)

    return p0_feature, pi_feature_120_1, pi_feature_1_1

# Assuming `x` is the input time series
# You can replace `x` with your actual time series variable

p0_feature, pi_feature_120_1, pi_feature_1_1 = calculate_persistence_statistics(x)

print("P0(x) feature:", p0_feature)
print("Pi(R120,1(x)) feature:", pi_feature_120_1)
print("Pi(R1,1(x)) feature:", pi_feature_1_1)
