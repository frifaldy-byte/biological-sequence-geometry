
# ============================================================
# BIOLOGICAL SEQUENCE GEOMETRY
# PHASE 2
# Sensitivity, Robustness and Higher-Dimensional Analysis
#
# All parameters are illustrative and dimensionless.
# No experimental data are used or fitted.
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


# ============================================================
# 1. GENERAL SETTINGS
# ============================================================

OUTPUT_DIR = "biological_sequence_geometry_phase2_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 20261117
N_MONTE_CARLO = 10000
PERTURBATION_RANGE = 0.20

# Consistent scientific figure formatting
plt.rcParams.update({
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300
})


# ============================================================
# 2. BASELINE MODEL PARAMETERS
# ============================================================

PARAMS = {
    "a1": 0.80,
    "a2": 0.65,
    "c1": 0.90,
    "c2": 1.10,
    "KI": 0.70,
    "KM": 0.90,
    "I0": 0.40,
    "M0": 0.30,
    "D1": 1.00,
    "D2": 1.00
}

A0 = 0.20

PARAMETER_NAMES = list(PARAMS.keys())


# ============================================================
# 3. ANALYTICAL SEQUENCE DIFFERENCE
# ============================================================

def sequence_difference(p):

    a1 = p["a1"]
    a2 = p["a2"]

    c1 = p["c1"]
    c2 = p["c2"]

    KI = p["KI"]
    KM = p["KM"]

    I0 = p["I0"]
    M0 = p["M0"]

    D1 = p["D1"]
    D2 = p["D2"]

    term_1 = (
        c2 * a1 * KI * D1 * D2
        / (
            (KI + I0)
            * (KI + I0 + a1 * D1)
        )
    )

    term_2 = (
        c1 * a2 * KM * D1 * D2
        / (
            (KM + M0)
            * (KM + M0 + a2 * D2)
        )
    )

    return term_1 - term_2


BASELINE_DELTA = sequence_difference(PARAMS)

if abs(BASELINE_DELTA) < 1e-12:
    raise RuntimeError(
        "Baseline sequence difference is too close to zero."
    )


# ============================================================
# 4. LOCAL PARAMETER SENSITIVITY
# ============================================================

def local_elasticity(
    p,
    parameter_name,
    relative_step=1e-5
):

    p_plus = p.copy()
    p_minus = p.copy()

    value = p[parameter_name]
    step = relative_step * value

    p_plus[parameter_name] = value + step
    p_minus[parameter_name] = value - step

    derivative = (
        sequence_difference(p_plus)
        - sequence_difference(p_minus)
    ) / (2.0 * step)

    return (
        value
        / sequence_difference(p)
    ) * derivative


ELASTICITIES = {
    name: local_elasticity(PARAMS, name)
    for name in PARAMETER_NAMES
}

sorted_sensitivity = sorted(
    ELASTICITIES.items(),
    key=lambda item: abs(item[1])
)

sensitivity_names = [
    item[0] for item in sorted_sensitivity
]

sensitivity_values = [
    item[1] for item in sorted_sensitivity
]


# Figure 1: Local sensitivity

fig, ax = plt.subplots(figsize=(9, 6))

ax.barh(
    sensitivity_names,
    sensitivity_values
)

ax.axvline(
    0,
    color="black",
    linewidth=0.9
)

ax.set_xlabel(
    "Local normalized sensitivity"
)

ax.set_ylabel(
    "Model parameter"
)

ax.set_title(
    "Local Sensitivity of the Sequence Difference",
    fontsize=14,
    fontweight="bold"
)

ax.grid(
    axis="x",
    alpha=0.20
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure1_local_sensitivity.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ============================================================
# 5. PARAMETER ROBUSTNESS ANALYSIS
# ============================================================

rng = np.random.default_rng(SEED)

sampled_differences = np.zeros(
    N_MONTE_CARLO
)

sampled_parameters = {
    name: np.zeros(N_MONTE_CARLO)
    for name in PARAMETER_NAMES
}

for i in range(N_MONTE_CARLO):

    sampled = {}

    for name, baseline_value in PARAMS.items():

        lower = (
            baseline_value
            * (1.0 - PERTURBATION_RANGE)
        )

        upper = (
            baseline_value
            * (1.0 + PERTURBATION_RANGE)
        )

        value = rng.uniform(lower, upper)

        sampled[name] = value

        sampled_parameters[name][i] = value

    sampled_differences[i] = sequence_difference(
        sampled
    )


positive_fraction = np.mean(
    sampled_differences > 0
)

negative_fraction = np.mean(
    sampled_differences < 0
)

sign_change_fraction = np.mean(
    np.sign(sampled_differences)
)

quantiles = np.quantile(
    sampled_differences,
    [
        0.025,
        0.25,
        0.50,
        0.75,
        0.975
    ]
)


# Figure 2: Robustness distribution

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.hist(
    sampled_differences,
    bins=55,
    density=True,
    alpha=0.80,
    edgecolor="black",
    linewidth=0.35
)

ax.axvline(
    BASELINE_DELTA,
    linestyle="--",
    linewidth=2,
    label="Baseline sequence difference"
)

ax.axvline(
    0,
    color="black",
    linewidth=1,
    label="No sequence effect"
)

ax.set_xlabel(
    r"Sequence Difference, $\Delta A$"
)

ax.set_ylabel(
    "Density"
)

ax.set_title(
    "Robustness Under Independent Parameter Perturbations",
    fontsize=14,
    fontweight="bold"
)

ax.legend()
ax.grid(alpha=0.15)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure2_robustness_distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ============================================================
# 6. PARAMETER ASSOCIATIONS
# ============================================================

correlations = {}

for name in PARAMETER_NAMES:

    correlations[name] = np.corrcoef(
        sampled_parameters[name],
        sampled_differences
    )[0, 1]


sorted_correlations = sorted(
    correlations.items(),
    key=lambda item: abs(item[1])
)

corr_names = [
    item[0] for item in sorted_correlations
]

corr_values = [
    item[1] for item in sorted_correlations
]


# Figure 3: Parameter correlations

fig, ax = plt.subplots(figsize=(9, 6))

ax.barh(
    corr_names,
    corr_values
)

ax.axvline(
    0,
    color="black",
    linewidth=0.9
)

ax.set_xlabel(
    "Pearson Correlation with Sequence Difference"
)

ax.set_ylabel(
    "Model Parameter"
)

ax.set_title(
    "Parameter Associations Across the Robustness Ensemble",
    fontsize=14,
    fontweight="bold"
)

ax.grid(
    axis="x",
    alpha=0.20
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure3_parameter_correlations.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ============================================================
# 7. FOUR-STATE CONTROL-AFFINE MODEL
# ============================================================

"""
Four-state system:

    x = (I, M, A, Z)

    dx/dt = u1 F1(x) + u2 F2(x)

Observable:

    H(x) = A + rho Z

The additional state Z represents a second adaptive
component. This is an illustrative mathematical extension.
"""

P4 = {
    "a1": 0.80,
    "a2": 0.65,
    "c1": 0.90,
    "c2": 1.10,
    "KI": 0.70,
    "KM": 0.90,
    "r1": 0.35,
    "r2": 0.25,
    "KZ": 0.80,
    "LZ": 1.00,
    "rho": 0.60
}


def F1(x):

    I, M, A, Z = x

    return np.array([
        P4["a1"],
        0.0,
        P4["c1"] * M / (P4["KM"] + M),
        P4["r1"] * M / (P4["KZ"] + M)
    ])


def F2(x):

    I, M, A, Z = x

    return np.array([
        0.0,
        P4["a2"],
        P4["c2"] * I / (P4["KI"] + I),
        P4["r2"] * I / (P4["LZ"] + I)
    ])


def observable(x):

    return (
        x[2]
        + P4["rho"] * x[3]
    )


def numerical_jacobian(
    vector_field,
    x,
    h=1e-6
):

    n = len(x)

    J = np.zeros((n, n))

    for j in range(n):

        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[j] += h
        x_minus[j] -= h

        J[:, j] = (
            vector_field(x_plus)
            - vector_field(x_minus)
        ) / (2.0 * h)

    return J


def lie_bracket(
    F1_function,
    F2_function,
    x
):

    J1 = numerical_jacobian(
        F1_function,
        x
    )

    J2 = numerical_jacobian(
        F2_function,
        x
    )

    # Convention:
    # [F1,F2] = DF2*F1 - DF1*F2

    return (
        J2 @ F1_function(x)
        - J1 @ F2_function(x)
    )


X0 = np.array([
    0.40,
    0.30,
    0.20,
    0.15
])

bracket_vector = lie_bracket(
    F1,
    F2,
    X0
)

grad_H = np.array([
    0.0,
    0.0,
    1.0,
    P4["rho"]
])

projected_bracket = float(
    grad_H @ bracket_vector
)


# ============================================================
# 8. NUMERICAL ORDER-OF-APPLICATION EXPERIMENT
# ============================================================

def apply_single_control(
    x_initial,
    vector_field,
    duration
):

    def rhs(t, x):
        return vector_field(x)

    result = solve_ivp(
        rhs,
        (0.0, duration),
        x_initial,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result.y[:, -1]


def apply_protocol_12(epsilon):

    after_first = apply_single_control(
        X0,
        F1,
        epsilon
    )

    after_second = apply_single_control(
        after_first,
        F2,
        epsilon
    )

    return after_second


def apply_protocol_21(epsilon):

    after_first = apply_single_control(
        X0,
        F2,
        epsilon
    )

    after_second = apply_single_control(
        after_first,
        F1,
        epsilon
    )

    return after_second


EPSILONS = np.geomspace(
    0.002,
    0.12,
    22
)

observed_differences = []
bracket_predictions = []
asymptotic_errors = []

for epsilon in EPSILONS:

    x12 = apply_protocol_12(epsilon)
    x21 = apply_protocol_21(epsilon)

    delta_H = (
        observable(x12)
        - observable(x21)
    )

    prediction = (
        epsilon**2
        * projected_bracket
    )

    observed_differences.append(delta_H)

    bracket_predictions.append(
        prediction
    )

    asymptotic_errors.append(
        abs(delta_H - prediction)
    )


observed_differences = np.array(
    observed_differences
)

bracket_predictions = np.array(
    bracket_predictions
)

asymptotic_errors = np.array(
    asymptotic_errors
)


# ============================================================
# 9. LIE-BRACKET VALIDATION
# ============================================================

bracket_ratios = (
    observed_differences
    / EPSILONS**2
)


# Figure 4: Higher-dimensional bracket

fig, ax = plt.subplots(figsize=(8.5, 5.5))

ax.semilogx(
    EPSILONS,
    bracket_ratios,
    "o-",
    label=r"Observed $\Delta H/\epsilon^2$"
)

ax.axhline(
    projected_bracket,
    linestyle="--",
    linewidth=2,
    label="Projected Lie bracket"
)

ax.set_xlabel(
    r"Pulse Duration, $\epsilon$"
)

ax.set_ylabel(
    r"$\Delta H/\epsilon^2$"
)

ax.set_title(
    "Higher-Dimensional Lie-Bracket Validation",
    fontsize=14,
    fontweight="bold"
)

ax.legend()
ax.grid(alpha=0.20)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure4_higher_dimensional_bracket.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ============================================================
# 10. ASYMPTOTIC ERROR ANALYSIS
# ============================================================

reference = (
    asymptotic_errors[0]
    * (EPSILONS / EPSILONS[0])**3
)


# Figure 5: Bracket approximation error

fig, ax = plt.subplots(figsize=(8.5, 5.5))

ax.loglog(
    EPSILONS,
    asymptotic_errors,
    "o-",
    label="Absolute bracket approximation error"
)

ax.loglog(
    EPSILONS,
    reference,
    "--",
    label=r"Reference Scaling: $O(\epsilon^3)$"
)

ax.set_xlabel(
    r"Pulse Duration, $\epsilon$"
)

ax.set_ylabel(
    "Absolute Approximation Error"
)

ax.set_title(
    "Asymptotic Error in the Four-State Model",
    fontsize=14,
    fontweight="bold"
)

ax.legend()
ax.grid(alpha=0.20)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure5_bracket_error.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ============================================================
# 11. AUTOMATED VALIDATION
# ============================================================

for name, value in PARAMS.items():

    assert value > 0, (
        f"Non-positive parameter: {name}"
    )


assert np.isfinite(BASELINE_DELTA)

assert np.all(
    np.isfinite(sampled_differences)
)

assert np.all(
    np.isfinite(observed_differences)
)

assert np.all(
    np.isfinite(asymptotic_errors)
)


small_epsilon_count = 5

small_epsilon_ratios = (
    bracket_ratios[:small_epsilon_count]
)

relative_bracket_errors = (
    np.abs(
        small_epsilon_ratios
        - projected_bracket
    )
    / max(
        abs(projected_bracket),
        1e-12
    )
)


assert relative_bracket_errors[0] < 0.05, (
    "Small-pulse Lie-bracket check did not meet "
    "the 5% relative-error tolerance."
)


# ============================================================
# 12. SAVE NUMERICAL SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "phase2_simulation_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "BIOLOGICAL SEQUENCE GEOMETRY\n"
    )

    file.write(
        "PHASE 2: SENSITIVITY AND ROBUSTNESS\n"
    )

    file.write("=" * 72 + "\n\n")

    file.write(
        "All parameters are illustrative and dimensionless.\n"
    )

    file.write(
        "No experimental data were used or fitted.\n\n"
    )

    file.write(
        "BASELINE PARAMETERS\n"
    )

    file.write("-" * 72 + "\n")

    for name, value in PARAMS.items():

        file.write(
            f"{name:10s}: {value:.8f}\n"
        )

    file.write(
        f"\nBaseline sequence difference: "
        f"{BASELINE_DELTA:.12f}\n\n"
    )

    file.write(
        "LOCAL NORMALIZED SENSITIVITIES\n"
    )

    file.write("-" * 72 + "\n")

    for name, value in sorted_sensitivity:

        file.write(
            f"{name:10s}: {value:+.8f}\n"
        )

    file.write(
        "\nROBUSTNESS ANALYSIS\n"
    )

    file.write("-" * 72 + "\n")

    file.write(
        f"Monte Carlo samples: {N_MONTE_CARLO}\n"
    )

    file.write(
        f"Independent parameter range: "
        f"+/- {100*PERTURBATION_RANGE:.1f}%\n"
    )

    file.write(
        f"Positive sequence difference: "
        f"{100*positive_fraction:.2f}%\n"
    )

    file.write(
        f"Negative sequence difference: "
        f"{100*negative_fraction:.2f}%\n"
    )

    file.write(
        f"Sign differs from baseline: "
        f"{100*sign_change_fraction:.2f}%\n"
    )

    file.write(
        "Quantiles (2.5%, 25%, 50%, 75%, 97.5%):\n"
    )

    file.write(
        np.array2string(
            quantiles,
            precision=10,
            separator=", "
        ) + "\n\n"
    )

    file.write(
        "FOUR-STATE LIE-BRACKET TEST\n"
    )

    file.write("-" * 72 + "\n")

    file.write(
        "Initial state (I, M, A, Z): "
        + np.array2string(
            X0,
            precision=8
        ) + "\n"
    )

    file.write(
        "Lie bracket vector: "
        + np.array2string(
            bracket_vector,
            precision=10
        ) + "\n"
    )

    file.write(
        f"Projected bracket: "
        f"{projected_bracket:.12f}\n"
    )

    file.write(
        f"Smallest-pulse observed ratio: "
        f"{bracket_ratios[0]:.12f}\n"
    )

    file.write(
        f"Smallest-pulse relative error: "
        f"{relative_bracket_errors[0]:.6e}\n\n"
    )

    file.write(
        "VALIDATION\n"
    )

    file.write("-" * 72 + "\n")

    file.write(
        "All parameter positivity checks passed.\n"
    )

    file.write(
        "All numerical finiteness checks passed.\n"
    )

    file.write(
        "Small-pulse Lie-bracket tolerance check passed.\n"
    )


# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

print("\nPHASE 2 COMPLETED")
print("=" * 65)

print("\nBASELINE")
print("-" * 65)

print(
    f"Baseline sequence difference: "
    f"{BASELINE_DELTA:.12f}"
)

print("\nLOCAL NORMALIZED SENSITIVITY")
print("-" * 65)

for name, value in sorted_sensitivity:

    print(
        f"{name:8s}: {value:+.6f}"
    )

print("\nROBUSTNESS ANALYSIS")
print("-" * 65)

print(
    f"Monte Carlo samples: {N_MONTE_CARLO}"
)

print(
    f"Positive sequence difference: "
    f"{100*positive_fraction:.2f}%"
)

print(
    f"Negative sequence difference: "
    f"{100*negative_fraction:.2f}%"
)

print(
    f"Sign differs from baseline: "
    f"{100*sign_change_fraction:.2f}%"
)

print("\nSequence-difference quantiles:")
print(quantiles)

print("\nFOUR-STATE LIE-BRACKET TEST")
print("-" * 65)

print(
    "Lie bracket vector:",
    bracket_vector
)

print(
    f"Projected bracket: "
    f"{projected_bracket:.12f}"
)

print(
    "Smallest-pulse observed ratio:",
    f"{bracket_ratios[0]:.12f}"
)

print(
    "Smallest-pulse relative error:",
    f"{relative_bracket_errors[0]:.6e}"
)

print("\nVALIDATION")
print("-" * 65)

print(
    "All parameter positivity checks passed."
)

print(
    "All numerical finiteness checks passed."
)

print(
    "Small-pulse Lie-bracket tolerance check passed."
)

print("\nOUTPUT FOLDER")
print("-" * 65)

print(
    os.path.abspath(OUTPUT_DIR)
)

print("\nGENERATED FILES")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(" -", filename)

print("\nAll Phase 2 calculations completed.")
