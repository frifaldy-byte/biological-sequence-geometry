
# ================================================================
# BIOLOGICAL SEQUENCE GEOMETRY
# Phase 1: Dose-matched stimulus sequences and weighted-area analysis
#
# Model:
#   dI/dt = a1*u1(t)
#   dM/dt = a2*u2(t)
#   dA/dt = c1*u1(t)*M/(KM + M)
#         + c2*u2(t)*I/(KI + I)
#
# All parameters are illustrative and dimensionless.
# They are not fitted to experimental measurements.
#
# Outputs:
#   Figure 1: Stimulus paths and adaptive response
#   Figure 2: Weighted sensitivity kernel
#   Figure 3: Sequence-equivalence landscape
#   Figure 4: Lie-bracket asymptotic validation
#   Figure 5: Dose-matched path-shape comparison
#
# ================================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# ================================================================
# 1. OUTPUT DIRECTORY AND PARAMETERS
# ================================================================

OUTPUT_DIR = Path("biological_sequence_geometry_phase1_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMS = {
    "a1": 0.80,
    "a2": 0.65,
    "c1": 0.90,
    "c2": 1.10,
    "KI": 0.70,
    "KM": 0.90,
    "I0": 0.40,
    "M0": 0.30,
    "A0": 0.20,
    "D1": 1.00,
    "D2": 1.00,
}

a1 = PARAMS["a1"]
a2 = PARAMS["a2"]
c1 = PARAMS["c1"]
c2 = PARAMS["c2"]
KI = PARAMS["KI"]
KM = PARAMS["KM"]

I0 = PARAMS["I0"]
M0 = PARAMS["M0"]
A0 = PARAMS["A0"]

D1 = PARAMS["D1"]
D2 = PARAMS["D2"]

N_STEPS = 20_000

# Parameter checks
if not all(v > 0 for v in [a1, a2, c1, c2, KI, KM, D1, D2]):
    raise ValueError("Rate, saturation, and dose parameters must be positive.")

if min(I0, M0, A0) < 0:
    raise ValueError("Initial states must be nonnegative.")


# ================================================================
# 2. MATHEMATICAL FUNCTIONS
# ================================================================

def weighted_kernel(p, q, I_init=I0, M_init=M0):
    """
    Signed weighted-area kernel.

    K(p,q) = dQ/dp - dP/dq
    """
    term_1 = (
        c2 * a1 * KI
        / (KI + I_init + a1 * p)**2
    )

    term_2 = (
        c1 * a2 * KM
        / (KM + M_init + a2 * q)**2
    )

    return term_1 - term_2


def exact_sequence_difference(
    dose1, dose2, I_init=I0, M_init=M0
):
    """
    Exact difference A_12 - A_21.

    Protocol 12: stimulus 1 followed by stimulus 2.
    Protocol 21: stimulus 2 followed by stimulus 1.

    Both protocols have identical cumulative doses.
    """

    increment_12 = (
        c1 * dose1 * M_init / (KM + M_init)
        + c2 * dose2 * (I_init + a1 * dose1)
        / (KI + I_init + a1 * dose1)
    )

    increment_21 = (
        c2 * dose2 * I_init / (KI + I_init)
        + c1 * dose1 * (M_init + a2 * dose2)
        / (KM + M_init + a2 * dose2)
    )

    return increment_12 - increment_21


def exact_final_A_protocol_12(dose1=D1, dose2=D2):
    """Exact final adaptive response for protocol 1 -> 2."""

    increment = (
        c1 * dose1 * M0 / (KM + M0)
        + c2 * dose2 * (I0 + a1 * dose1)
        / (KI + I0 + a1 * dose1)
    )

    return A0 + increment


def exact_final_A_protocol_21(dose1=D1, dose2=D2):
    """Exact final adaptive response for protocol 2 -> 1."""

    increment = (
        c2 * dose2 * I0 / (KI + I0)
        + c1 * dose1 * (M0 + a2 * dose2)
        / (KM + M0 + a2 * dose2)
    )

    return A0 + increment


# ================================================================
# 3. STIMULUS PROTOCOLS
# ================================================================

def path_protocol_12(t):
    """Stimulus 1 first, followed by stimulus 2."""

    if t <= 0.5:
        return 2.0 * t * D1, 0.0

    return D1, 2.0 * (t - 0.5) * D2


def path_protocol_21(t):
    """Stimulus 2 first, followed by stimulus 1."""

    if t <= 0.5:
        return 0.0, 2.0 * t * D2

    return 2.0 * (t - 0.5) * D1, D2


def path_simultaneous(t):
    """Both stimuli are applied simultaneously."""

    return t * D1, t * D2


def path_alternating(t, n_switches=20):
    """
    Alternating staircase protocol.

    The sequence is:
        stimulus 1, stimulus 2, stimulus 1, stimulus 2, ...

    Each input receives its complete prescribed dose.
    """

    if not 0.0 <= t <= 1.0:
        raise ValueError("Time must be between 0 and 1.")

    if t >= 1.0:
        return D1, D2

    n_segments = 2 * n_switches
    scaled_time = t * n_segments

    # Correct for floating-point values close to switching boundaries.
    nearest_boundary = round(scaled_time)

    if abs(scaled_time - nearest_boundary) < 1e-10:
        scaled_time = float(nearest_boundary)

    segment_index = min(
        int(np.floor(scaled_time)),
        n_segments - 1
    )

    fraction = scaled_time - segment_index
    fraction = min(max(fraction, 0.0), 1.0)

    if segment_index % 2 == 0:

        p = (
            (segment_index // 2)
            * D1 / n_switches
        )

        q = (
            (segment_index // 2)
            * D2 / n_switches
        )

        p += fraction * D1 / n_switches

    else:

        p = (
            ((segment_index + 1) // 2)
            * D1 / n_switches
        )

        q = (
            (segment_index // 2)
            * D2 / n_switches
        )

        q += fraction * D2 / n_switches

    return p, q


PROTOCOLS = {
    "Stimulus 1 then 2": path_protocol_12,
    "Stimulus 2 then 1": path_protocol_21,
    "Simultaneous": path_simultaneous,
    "Alternating": path_alternating,
}


# ================================================================
# 4. NUMERICAL LINE INTEGRATION
# ================================================================

def integrate_exposure_path(path_function, n_steps=N_STEPS):
    """
    Evaluate:

        dA = P(p,q) dp + Q(p,q) dq

    using the composite midpoint rule.

    The biological states I and M are recovered from
    cumulative exposures.
    """

    if n_steps < 2:
        raise ValueError("n_steps must be at least 2.")

    time = np.linspace(0.0, 1.0, n_steps + 1)

    path_values = np.array(
        [path_function(t) for t in time],
        dtype=float
    )

    p = path_values[:, 0]
    q = path_values[:, 1]

    # Cumulative exposure must never decrease.
    if np.any(np.diff(p) < -1e-12):
        raise ValueError("Inflammatory exposure is not monotonic.")

    if np.any(np.diff(q) < -1e-12):
        raise ValueError("Metabolic exposure is not monotonic.")

    A = np.empty(n_steps + 1)
    A[0] = A0

    for j in range(1, n_steps + 1):

        dp = p[j] - p[j - 1]
        dq = q[j] - q[j - 1]

        p_mid = 0.5 * (p[j] + p[j - 1])
        q_mid = 0.5 * (q[j] + q[j - 1])

        P = (
            c1 * (M0 + a2 * q_mid)
            / (KM + M0 + a2 * q_mid)
        )

        Q = (
            c2 * (I0 + a1 * p_mid)
            / (KI + I0 + a1 * p_mid)
        )

        A[j] = A[j - 1] + P * dp + Q * dq

    I = I0 + a1 * p
    M = M0 + a2 * q

    return {
        "time": time,
        "p": p,
        "q": q,
        "I": I,
        "M": M,
        "A": A,
    }


# Run all four protocols.
SIMULATION_RESULTS = {
    name: integrate_exposure_path(path)
    for name, path in PROTOCOLS.items()
}


# ================================================================
# 5. FIGURE 1 — STIMULUS PATHS AND RESPONSE TRAJECTORIES
# ================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

for name, result in SIMULATION_RESULTS.items():

    axes[0].plot(
        result["p"],
        result["q"],
        linewidth=2,
        label=name
    )

    axes[1].plot(
        result["time"],
        result["A"],
        linewidth=2,
        label=name
    )

axes[0].set_title(
    "Dose-matched stimulus paths",
    fontweight="bold"
)

axes[0].set_xlabel("Cumulative inflammatory exposure, p")
axes[0].set_ylabel("Cumulative metabolic exposure, q")
axes[0].legend(frameon=False, fontsize=8)
axes[0].grid(False)

axes[1].set_title(
    "Adaptive response during stimulation",
    fontweight="bold"
)

axes[1].set_xlabel("Normalized protocol time")
axes[1].set_ylabel("Adaptive response, A(t)")
axes[1].legend(frameon=False, fontsize=8)
axes[1].grid(False)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "figure1_paths_and_response.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ================================================================
# 6. FIGURE 2 — WEIGHTED SENSITIVITY KERNEL
# ================================================================

p_grid = np.linspace(0.0, D1, 250)
q_grid = np.linspace(0.0, D2, 250)

P_GRID, Q_GRID = np.meshgrid(p_grid, q_grid)

K_GRID = weighted_kernel(P_GRID, Q_GRID)

fig, ax = plt.subplots(figsize=(7.2, 5.6))

contour = ax.contourf(
    P_GRID,
    Q_GRID,
    K_GRID,
    levels=35
)

zero_contour = ax.contour(
    P_GRID,
    Q_GRID,
    K_GRID,
    levels=[0.0],
    linewidths=1.6
)

ax.clabel(
    zero_contour,
    fmt={0.0: "K = 0"},
    inline=True,
    fontsize=9
)

fig.colorbar(
    contour,
    ax=ax,
    label="Weighted sensitivity kernel, K(p,q)"
)

ax.set_title(
    "Weighted sequence-sensitivity kernel",
    fontweight="bold"
)

ax.set_xlabel("Cumulative inflammatory exposure, p")
ax.set_ylabel("Cumulative metabolic exposure, q")
ax.grid(False)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "figure2_weighted_kernel.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ================================================================
# 7. FIGURE 3 — SEQUENCE-EQUIVALENCE LANDSCAPE
# ================================================================

iv = np.linspace(0.05, 1.8, 220)
mv = np.linspace(0.05, 1.8, 220)

II, MM = np.meshgrid(iv, mv)

DA = (
    c2 * a1 * KI * D1 * D2
    / ((KI + II) * (KI + II + a1 * D1))
    - c1 * a2 * KM * D1 * D2
    / ((KM + MM) * (KM + MM + a2 * D2))
)

fig, ax = plt.subplots(figsize=(7.2, 5.6))

lim = np.max(np.abs(DA))

im = ax.contourf(
    II,
    MM,
    DA,
    levels=np.linspace(-lim, lim, 35),
    cmap="coolwarm"
)

cs = ax.contour(
    II,
    MM,
    DA,
    levels=[0.0],
    colors="black",
    linewidths=1.8
)

ax.clabel(
    cs,
    fmt={0.0: r"$\Delta A=0$"},
    inline=True,
    fontsize=9
)

# Baseline initial state: exactly one marker
ax.scatter(
    [I0],
    [M0],
    s=75,
    c="black",
    marker="x",
    linewidths=2.0,
    label=r"Baseline $(I_0,M_0)=(0.40,0.30)$",
    zorder=10
)

fig.colorbar(
    im,
    ax=ax,
    label=r"$\Delta A=A_{12}-A_{21}$"
)

ax.set_title(
    "Sequence-equivalence landscape",
    fontweight="bold"
)

ax.set_xlabel(r"Initial inflammatory state, $I_0$")
ax.set_ylabel(r"Initial metabolic state, $M_0$")

ax.set_xlim(0.05, 1.8)
ax.set_ylim(0.05, 1.8)

ax.legend(
    frameon=False,
    loc="lower right"
)

ax.grid(False)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "figure3_equivalence_landscape.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ================================================================
# 8. FIGURE 4 — LIE-BRACKET ASYMPTOTIC VALIDATION
# ================================================================

epsilon_values = np.logspace(-3.0, -0.35, 30)

K_initial = weighted_kernel(0.0, 0.0)

exact_differences = np.array([
    exact_sequence_difference(eps, eps)
    for eps in epsilon_values
])

second_order_approximations = (
    epsilon_values**2 * K_initial
)

asymptotic_residuals = np.abs(
    exact_differences - second_order_approximations
)

# Estimate the slope over the first ten small-pulse values.
slope_estimate = np.polyfit(
    np.log(epsilon_values[:10]),
    np.log(asymptotic_residuals[:10]),
    deg=1
)[0]

reference_line = (
    asymptotic_residuals[5]
    * (epsilon_values / epsilon_values[5])**3
)

fig, ax = plt.subplots(figsize=(7.2, 5.4))

ax.loglog(
    epsilon_values,
    asymptotic_residuals,
    "o-",
    label="Absolute asymptotic residual"
)

ax.loglog(
    epsilon_values,
    reference_line,
    "--",
    label="Reference slope: O(ε³)"
)

ax.set_title(
    "Validation of the Lie-bracket approximation",
    fontweight="bold"
)

ax.set_xlabel("Pulse duration, ε")
ax.set_ylabel("|ΔA − ε²K(0,0)|")
ax.legend(frameon=False)
ax.grid(False)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "figure4_lie_bracket_validation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ================================================================
# 9. FIGURE 5 — DOSE-MATCHED PATH-SHAPE COMPARISON
# ================================================================

def power_path_response_increment(exponent):
    """
    Exposure path:

        p(t) = D1*t
        q(t) = D2*t**exponent

    Every positive exponent has the same endpoint (D1,D2).
    """

    if exponent <= 0:
        raise ValueError("The path exponent must be positive.")

    def integrand(t):

        p = D1 * t
        q = D2 * t**exponent

        dp_dt = D1

        if t == 0.0:
            # Endpoint value does not affect the integral.
            dq_dt = 0.0
        else:
            dq_dt = (
                D2 * exponent
                * t**(exponent - 1.0)
            )

        P = (
            c1 * (M0 + a2 * q)
            / (KM + M0 + a2 * q)
        )

        Q = (
            c2 * (I0 + a1 * p)
            / (KI + I0 + a1 * p)
        )

        return P * dp_dt + Q * dq_dt

    value, estimated_error = quad(
        integrand,
        0.0,
        1.0,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=300
    )

    return value, estimated_error


path_exponents = [0.5, 1.0, 2.0, 4.0]

path_shape_results = [
    power_path_response_increment(r)
    for r in path_exponents
]

path_shape_increments = [
    result[0] for result in path_shape_results
]

simultaneous_increment = path_shape_increments[
    path_exponents.index(1.0)
]

path_labels = [
    f"r = {r:g}"
    for r in path_exponents
]

fig, ax = plt.subplots(figsize=(7.2, 5.4))

ax.plot(
    path_labels,
    path_shape_increments,
    "o-",
    linewidth=2,
    label="Path-integrated response"
)

ax.axhline(
    simultaneous_increment,
    linestyle="--",
    linewidth=1.5,
    label="Simultaneous-path reference"
)

ax.set_title(
    "Endpoint response across dose-matched path shapes",
    fontweight="bold"
)

ax.set_xlabel("Exposure-path family: q(t) = tʳ")
ax.set_ylabel("Final adaptive response increment, A(T) − A₀")
ax.legend(frameon=False)
ax.grid(False)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "figure5_path_shape_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ================================================================
# 10. INDEPENDENT WEIGHTED-AREA VERIFICATION
# ================================================================

# Green's theorem:
#
# A_12 - A_21 =
# integral_0^D1 integral_0^D2 K(p,q) dq dp
#
# The closed curve is formed by protocol 1->2 followed by
# the reverse of protocol 2->1. Its orientation is positive.

def inner_area_integral(p):

    value, _ = quad(
        lambda q: weighted_kernel(p, q),
        0.0,
        D2,
        epsabs=1e-12,
        epsrel=1e-12
    )

    return value


area_integral, area_quadrature_error = quad(
    inner_area_integral,
    0.0,
    D1,
    epsabs=1e-12,
    epsrel=1e-12
)

analytical_delta = exact_sequence_difference(D1, D2)

numerical_A12 = (
    SIMULATION_RESULTS["Stimulus 1 then 2"]["A"][-1]
)

numerical_A21 = (
    SIMULATION_RESULTS["Stimulus 2 then 1"]["A"][-1]
)

numerical_delta = numerical_A12 - numerical_A21

exact_A12 = exact_final_A_protocol_12(D1, D2)
exact_A21 = exact_final_A_protocol_21(D1, D2)

error_A12 = abs(numerical_A12 - exact_A12)
error_A21 = abs(numerical_A21 - exact_A21)

area_identity_error = abs(
    analytical_delta - area_integral
)


# ================================================================
# 11. NUMERICAL RESULTS AND DIAGNOSTICS
# ================================================================

print("\nBIOLOGICAL SEQUENCE GEOMETRY")
print("PHASE 1: NUMERICAL SIMULATIONS")
print("-" * 68)

print("All parameters are illustrative and dimensionless.")
print("No experimental data were used or fitted.\n")

print(
    f"{'Protocol':<25}"
    f"{'Final I':>11}"
    f"{'Final M':>11}"
    f"{'Final A':>13}"
)

print("-" * 68)

for name, result in SIMULATION_RESULTS.items():

    print(
        f"{name:<25}"
        f"{result['I'][-1]:>11.6f}"
        f"{result['M'][-1]:>11.6f}"
        f"{result['A'][-1]:>13.6f}"
    )


print("\nANALYTICAL AND NUMERICAL CHECKS")
print("-" * 68)

print(f"Exact A_12:                 {exact_A12:.12f}")
print(f"Numerical A_12:             {numerical_A12:.12f}")
print(f"Error, protocol 1 -> 2:     {error_A12:.3e}")

print()

print(f"Exact A_21:                 {exact_A21:.12f}")
print(f"Numerical A_21:             {numerical_A21:.12f}")
print(f"Error, protocol 2 -> 1:     {error_A21:.3e}")

print()

print(f"Exact sequence difference: {analytical_delta:.12f}")
print(f"Numerical path difference: {numerical_delta:.12f}")
print(f"Weighted-area integral:    {area_integral:.12f}")
print(f"Area identity discrepancy: {area_identity_error:.3e}")
print(f"Quadrature error estimate: {area_quadrature_error:.3e}")

print()

print(
    "Estimated small-pulse residual slope: "
    f"{slope_estimate:.4f}"
)

print("Expected asymptotic slope: approximately 3")


print("\nPATH-SHAPE RESULTS")
print("-" * 68)

for exponent, result in zip(
    path_exponents,
    path_shape_results
):

    value, error = result

    print(
        f"q(t)=t^{exponent:<4g}"
        f"  response increment={value:.10f}"
        f"  quadrature estimate={error:.2e}"
    )


# ================================================================
# 12. AUTOMATIC VALIDATION TESTS
# ================================================================

# All protocols must end at the same cumulative doses.
for name, result in SIMULATION_RESULTS.items():

    assert np.isclose(result["p"][-1], D1, atol=1e-10), (
        f"Incorrect inflammatory dose in {name}"
    )

    assert np.isclose(result["q"][-1], D2, atol=1e-10), (
        f"Incorrect metabolic dose in {name}"
    )

    assert np.all(np.diff(result["p"]) >= -1e-12), (
        f"Inflammatory exposure decreased in {name}"
    )

    assert np.all(np.diff(result["q"]) >= -1e-12), (
        f"Metabolic exposure decreased in {name}"
    )

# The numerical integration must agree with the exact expressions.
assert error_A12 < 1e-8
assert error_A21 < 1e-8

# The weighted-area identity must agree with the exact formula.
assert area_identity_error < 1e-9

# The asymptotic slope should be close to three.
assert 2.8 < slope_estimate < 3.2

print("\nVALIDATION")
print("-" * 68)
print("All cumulative-dose checks passed.")
print("All monotonicity checks passed.")
print("Exact endpoint comparisons passed.")
print("Weighted-area identity check passed.")
print("Small-pulse asymptotic check passed.")


# ================================================================
# 13. SAVE PARAMETER AND RESULT SUMMARY
# ================================================================

summary_path = OUTPUT_DIR / "simulation_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("BIOLOGICAL SEQUENCE GEOMETRY — PHASE 1\n")
    file.write("Illustrative dimensionless model\n")
    file.write("No experimental data used or fitted.\n\n")

    file.write("PARAMETERS\n")

    for key, value in PARAMS.items():
        file.write(f"{key} = {value}\n")

    file.write("\nENDPOINT RESULTS\n")

    for name, result in SIMULATION_RESULTS.items():

        file.write(
            f"{name}: "
            f"I={result['I'][-1]:.12f}, "
            f"M={result['M'][-1]:.12f}, "
            f"A={result['A'][-1]:.12f}\n"
        )

    file.write("\nMATHEMATICAL CHECKS\n")
    file.write(f"Exact A12 = {exact_A12:.12f}\n")
    file.write(f"Numerical A12 = {numerical_A12:.12f}\n")
    file.write(f"Exact A21 = {exact_A21:.12f}\n")
    file.write(f"Numerical A21 = {numerical_A21:.12f}\n")
    file.write(f"Exact Delta A = {analytical_delta:.12f}\n")
    file.write(f"Area integral = {area_integral:.12f}\n")
    file.write(
        f"Area identity discrepancy = "
        f"{area_identity_error:.3e}\n"
    )
    file.write(
        f"Estimated asymptotic slope = "
        f"{slope_estimate:.6f}\n"
    )

print("\nOUTPUT FILES")
print("-" * 68)
print(f"Folder: {OUTPUT_DIR.resolve()}")

for filename in sorted(OUTPUT_DIR.glob("*.png")):
    print(" -", filename.name)

print(" -", summary_path.name)

print("\nSimulation completed successfully.")
