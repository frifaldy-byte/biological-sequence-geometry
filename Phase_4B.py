
# ============================================================
# BIOLOGICAL SEQUENCE GEOMETRY
# PHASE 4B: HIGHER-ORDER PATH DEPENDENCE
#
# Equal dose
# Equal endpoints
# Equal weighted-area integral
# Different higher-order path response
#
# Numerical integration: scipy.solve_ivp (DOP853)
# All quantities are illustrative and dimensionless.
# No experimental data are used or fitted.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


# ============================================================
# 2. OUTPUT DIRECTORY AND NUMERICAL SETTINGS
# ============================================================

OUTPUT_DIR = "biological_sequence_geometry_phase4B_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RTOL = 1e-12
ATOL = 1e-14

POINTS_PER_SEGMENT = 80


# ============================================================
# 3. PUBLICATION-STYLE GRAPH SETTINGS
# ============================================================

COLOR_A = "#34443D"       # Deep charcoal green
COLOR_B = "#C17B5B"       # Terracotta
COLOR_SAGE = "#648579"    # Muted sage
COLOR_OCHRE = "#B5A36A"   # Muted ochre
COLOR_TEXT = "#303632"
COLOR_GRID = "#D9DEDA"
COLOR_BACKGROUND = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelcolor": COLOR_TEXT,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.color": COLOR_TEXT,
    "axes.edgecolor": "#7C8580",
    "axes.linewidth": 0.8,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.facecolor": COLOR_BACKGROUND,
    "figure.facecolor": COLOR_BACKGROUND
})


def style_axes(ax, grid_axis="both"):

    ax.set_facecolor(COLOR_BACKGROUND)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#7C8580")
    ax.spines["bottom"].set_color("#7C8580")

    ax.tick_params(
        axis="both",
        colors=COLOR_TEXT,
        labelsize=9
    )

    ax.xaxis.label.set_color(COLOR_TEXT)
    ax.yaxis.label.set_color(COLOR_TEXT)

    ax.grid(
        axis=grid_axis,
        color=COLOR_GRID,
        linewidth=0.7,
        alpha=0.65
    )

    ax.set_axisbelow(True)


# ============================================================
# 4. MATHEMATICAL MODEL
# ============================================================

"""
State vector:

    x = (p, q, z1, z2)

Dynamics:

    dp/dt  = u1
    dq/dt  = u2
    dz1/dt = p*u2
    dz2/dt = p^2*u2

Integral representations:

    z1(T) = integral_gamma p dq

    z2(T) = integral_gamma p^2 dq

Control-affine representation:

    dx/dt = u1*F1(x) + u2*F2(x)

where:

    F1 = (1, 0, 0, 0)

    F2 = (0, 1, p, p^2)
"""


def F1(x):

    return np.array([
        1.0,
        0.0,
        0.0,
        0.0
    ])


def F2(x):

    p, q, z1, z2 = x

    return np.array([
        0.0,
        1.0,
        p,
        p**2
    ])


def rhs(t, x, u1, u2):

    return (
        u1 * F1(x)
        + u2 * F2(x)
    )


# ============================================================
# 5. DEFINE DOSE-MATCHED PROTOCOLS
# ============================================================

"""
Each segment is represented by:

    (increment in p, increment in q)

Each segment has unit duration.

Protocol A:

    (0.25, 0)
    (0, 0.50)
    (0.50, 0)
    (0, 0.50)
    (0.25, 0)

Protocol B:

    (0.50, 0)
    (0, 1.00)
    (0.50, 0)

Both protocols have:

    Final p = 1
    Final q = 1

Analytical results:

    Protocol A:
        z1 = 0.5000
        z2 = 0.3125

    Protocol B:
        z1 = 0.5000
        z2 = 0.2500
"""

PROTOCOL_A = [
    (0.25, 0.00),
    (0.00, 0.50),
    (0.50, 0.00),
    (0.00, 0.50),
    (0.25, 0.00)
]

PROTOCOL_B = [
    (0.50, 0.00),
    (0.00, 1.00),
    (0.50, 0.00)
]


# ============================================================
# 6. NUMERICAL INTEGRATION FUNCTION
# ============================================================

def simulate_protocol(
    protocol,
    label,
    points_per_segment=POINTS_PER_SEGMENT
):

    x = np.zeros(4)

    history = {
        "time": [],
        "p": [],
        "q": [],
        "z1": [],
        "z2": []
    }

    global_time = 0.0

    for dp, dq in protocol:

        u1 = float(dp)
        u2 = float(dq)

        if u1 < 0 or u2 < 0:

            raise ValueError(
                "Stimulus controls must be nonnegative."
            )

        def local_rhs(t, state):

            return rhs(
                t,
                state,
                u1,
                u2
            )

        evaluation_times = np.linspace(
            0.0,
            1.0,
            points_per_segment
        )

        solution = solve_ivp(
            local_rhs,
            (0.0, 1.0),
            x,
            method="DOP853",
            t_eval=evaluation_times,
            rtol=RTOL,
            atol=ATOL
        )

        if not solution.success:

            raise RuntimeError(
                f"Integration failed for {label}: "
                f"{solution.message}"
            )

        start_index = (
            0 if len(history["time"]) == 0 else 1
        )

        for j in range(
            start_index,
            len(solution.t)
        ):

            history["time"].append(
                global_time + solution.t[j]
            )

            history["p"].append(
                solution.y[0, j]
            )

            history["q"].append(
                solution.y[1, j]
            )

            history["z1"].append(
                solution.y[2, j]
            )

            history["z2"].append(
                solution.y[3, j]
            )

        x = solution.y[:, -1]

        global_time += 1.0

    for key in history:

        history[key] = np.asarray(
            history[key]
        )

    return {
        "label": label,
        "final_state": x,
        "history": history,
        "duration": global_time
    }


# ============================================================
# 7. RUN BOTH PROTOCOLS
# ============================================================

result_A = simulate_protocol(
    PROTOCOL_A,
    "Protocol A"
)

result_B = simulate_protocol(
    PROTOCOL_B,
    "Protocol B"
)

X_A = result_A["final_state"]
X_B = result_B["final_state"]


# ============================================================
# 8. ANALYTICAL REFERENCE SOLUTIONS
# ============================================================

ANALYTICAL_A = np.array([
    1.0000,
    1.0000,
    0.5000,
    0.3125
])

ANALYTICAL_B = np.array([
    1.0000,
    1.0000,
    0.5000,
    0.2500
])

ANALYTICAL_HIGHER_ORDER_DIFFERENCE = 0.0625


# ============================================================
# 9. NUMERICAL ERROR AND CONSISTENCY CHECKS
# ============================================================

NUMERICAL_DIFFERENCE = X_A - X_B

DOSE_DIFFERENCE = NUMERICAL_DIFFERENCE[:2]

AREA_DIFFERENCE = X_A[2] - X_B[2]

HIGHER_ORDER_DIFFERENCE = X_A[3] - X_B[3]

ENDPOINT_ERROR_A = np.max(
    np.abs(X_A - ANALYTICAL_A)
)

ENDPOINT_ERROR_B = np.max(
    np.abs(X_B - ANALYTICAL_B)
)

DOSE_ERROR = np.max(
    np.abs(DOSE_DIFFERENCE)
)

AREA_ERROR = abs(
    AREA_DIFFERENCE
)

HIGHER_ORDER_ERROR = abs(
    HIGHER_ORDER_DIFFERENCE
    - ANALYTICAL_HIGHER_ORDER_DIFFERENCE
)


# ============================================================
# 10. LIE-BRACKET CALCULATIONS
# ============================================================

"""
Bracket convention:

    [F1,F2] = DF2*F1 - DF1*F2

For the present system:

    [F1,F2] = (0,0,1,2p)

Nested bracket:

    [F1,[F1,F2]] = (0,0,0,2)

The second-order bracket acts on z1.

The nested third-order bracket acts on z2.
"""


def bracket_F1_F2(x):

    p, q, z1, z2 = x

    return np.array([
        0.0,
        0.0,
        1.0,
        2.0 * p
    ])


def nested_bracket_F1_F1F2(x):

    return np.array([
        0.0,
        0.0,
        0.0,
        2.0
    ])


INITIAL_STATE = np.zeros(4)

BRACKET_SECOND = bracket_F1_F2(
    INITIAL_STATE
)

BRACKET_THIRD = nested_bracket_F1_F1F2(
    INITIAL_STATE
)

BRACKET_SECOND_NORM = np.linalg.norm(
    BRACKET_SECOND
)

BRACKET_THIRD_NORM = np.linalg.norm(
    BRACKET_THIRD
)


# ============================================================
# 11. VALIDATION
# ============================================================

assert np.allclose(
    X_A,
    ANALYTICAL_A,
    rtol=0,
    atol=1e-10
)

assert np.allclose(
    X_B,
    ANALYTICAL_B,
    rtol=0,
    atol=1e-10
)

assert np.allclose(
    X_A[:2],
    X_B[:2],
    rtol=0,
    atol=1e-10
)

assert np.isclose(
    X_A[2],
    X_B[2],
    rtol=0,
    atol=1e-10
)

assert np.isclose(
    HIGHER_ORDER_DIFFERENCE,
    ANALYTICAL_HIGHER_ORDER_DIFFERENCE,
    rtol=0,
    atol=1e-10
)

assert np.all(np.isfinite(X_A))
assert np.all(np.isfinite(X_B))

assert np.all(X_A >= 0)
assert np.all(X_B >= 0)

assert np.isclose(
    BRACKET_SECOND_NORM,
    1.0
)

assert np.isclose(
    BRACKET_THIRD_NORM,
    2.0
)


# ============================================================
# 12. FIGURE 1: DOSE-MATCHED PATHS
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 6.5),
    facecolor=COLOR_BACKGROUND
)

ax.plot(
    result_A["history"]["p"],
    result_A["history"]["q"],
    color=COLOR_A,
    linewidth=2.5,
    label="Protocol A"
)

ax.plot(
    result_B["history"]["p"],
    result_B["history"]["q"],
    color=COLOR_B,
    linewidth=2.5,
    linestyle="--",
    label="Protocol B"
)

ax.scatter(
    [0, 1],
    [0, 1],
    color=COLOR_TEXT,
    s=45,
    zorder=5
)

ax.set_xlabel(
    r"Cumulative stimulus 1 exposure, $p$"
)

ax.set_ylabel(
    r"Cumulative stimulus 2 exposure, $q$"
)

ax.set_title(
    "Dose-Matched Paths with Identical Endpoints",
    fontweight="bold",
    pad=14
)

ax.set_xlim(-0.04, 1.04)
ax.set_ylim(-0.04, 1.04)

ax.set_aspect("equal", adjustable="box")

style_axes(ax, "both")

ax.legend(
    frameon=False,
    loc="upper left"
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure1_dose_matched_paths.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ============================================================
# 13. FIGURE 2: PATH-DEPENDENT STATE VARIABLES
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5.2),
    facecolor=COLOR_BACKGROUND
)

# z1
axes[0].plot(
    result_A["history"]["time"],
    result_A["history"]["z1"],
    color=COLOR_A,
    linewidth=2.3,
    label="Protocol A"
)

axes[0].plot(
    result_B["history"]["time"],
    result_B["history"]["z1"],
    color=COLOR_B,
    linewidth=2.3,
    linestyle="--",
    label="Protocol B"
)

axes[0].set_xlabel("Protocol time")

axes[0].set_ylabel(
    r"$z_1=\int_\gamma p\,dq$"
)

axes[0].set_title(
    "Weighted-Area Accumulation",
    fontweight="bold",
    pad=12
)

style_axes(axes[0], "both")

axes[0].legend(frameon=False)


# z2
axes[1].plot(
    result_A["history"]["time"],
    result_A["history"]["z2"],
    color=COLOR_SAGE,
    linewidth=2.5,
    label="Protocol A"
)

axes[1].plot(
    result_B["history"]["time"],
    result_B["history"]["z2"],
    color=COLOR_OCHRE,
    linewidth=2.5,
    linestyle="--",
    label="Protocol B"
)

axes[1].set_xlabel("Protocol time")

axes[1].set_ylabel(
    r"$z_2=\int_\gamma p^2\,dq$"
)

axes[1].set_title(
    "Higher-Order Path Accumulation",
    fontweight="bold",
    pad=12
)

style_axes(axes[1], "both")

axes[1].legend(frameon=False)

fig.suptitle(
    "Comparison of Path-Dependent State Variables",
    fontsize=14,
    fontweight="bold",
    color=COLOR_TEXT,
    y=1.02
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure2_path_dependent_states.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ============================================================
# 14. FIGURE 3: ENDPOINT COMPARISON
# ============================================================

endpoint_names = [
    "Inflammatory exposure (p)",
    "Metabolic exposure (q)",
    "Area-related state (z1)",
    "Higher-order state (z2)"
]

values_A = X_A
values_B = X_B

y = np.arange(
    len(endpoint_names)
)

height = 0.31

fig, ax = plt.subplots(
    figsize=(10, 6.5),
    facecolor=COLOR_BACKGROUND
)

bars_A = ax.barh(
    y - height / 2,
    values_A,
    height=height,
    color=COLOR_A,
    edgecolor="white",
    linewidth=0.8,
    label="Protocol A"
)

bars_B = ax.barh(
    y + height / 2,
    values_B,
    height=height,
    color=COLOR_B,
    edgecolor="white",
    linewidth=0.8,
    label="Protocol B"
)

ax.set_yticks(y)
ax.set_yticklabels(endpoint_names)

ax.invert_yaxis()

ax.set_xlim(0, 1.15)

ax.set_xlabel(
    "Final state value"
)

ax.set_title(
    "Endpoint Comparison Across Dose-Matched Protocols",
    fontweight="bold",
    pad=15
)

for bars in [bars_A, bars_B]:

    for bar in bars:

        value = bar.get_width()

        ax.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            color=COLOR_TEXT
        )

style_axes(ax, "x")

ax.legend(
    frameon=False,
    loc="lower right",
    ncol=2
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure3_endpoint_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ============================================================
# 15. FIGURE 4: LIE-BRACKET STRUCTURE
# ============================================================

bracket_labels = [
    "Second-order bracket",
    "Third-order nested bracket"
]

bracket_values = [
    BRACKET_SECOND_NORM,
    BRACKET_THIRD_NORM
]

bracket_colors = [
    COLOR_SAGE,
    COLOR_OCHRE
]

y_bracket = np.arange(
    len(bracket_labels)
)

fig, ax = plt.subplots(
    figsize=(9, 5.5),
    facecolor=COLOR_BACKGROUND
)

bars = ax.barh(
    y_bracket,
    bracket_values,
    height=0.48,
    color=bracket_colors,
    edgecolor="white",
    linewidth=1.0
)

ax.set_yticks(y_bracket)
ax.set_yticklabels(bracket_labels)

ax.invert_yaxis()

ax.set_xlim(0, 2.45)

ax.set_xlabel(
    "Euclidean norm at the initial state"
)

ax.set_title(
    "Lie-Bracket Structure of the Higher-Order Model",
    fontweight="bold",
    pad=15
)

for bar in bars:

    value = bar.get_width()

    ax.text(
        value + 0.06,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=COLOR_TEXT
    )

style_axes(ax, "x")

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure4_lie_bracket_structure.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ============================================================
# 16. SAVE ENDPOINT DATA
# ============================================================

endpoint_data = np.column_stack([
    X_A,
    X_B,
    X_A - X_B
])

np.savetxt(
    os.path.join(
        OUTPUT_DIR,
        "endpoint_comparison.csv"
    ),
    endpoint_data,
    delimiter=",",
    header="Protocol_A,Protocol_B,Difference",
    comments=""
)


# ============================================================
# 17. SAVE SUMMARY REPORT
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "phase4B_simulation_summary.txt"
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
        "PHASE 4B: HIGHER-ORDER PATH DEPENDENCE\n"
    )

    file.write("=" * 75 + "\n\n")

    file.write(
        "All quantities are illustrative and dimensionless.\n"
    )

    file.write(
        "No experimental data were used or fitted.\n\n"
    )

    file.write(
        "PROTOCOL A FINAL STATE\n"
    )

    file.write("-" * 75 + "\n")

    file.write(
        np.array2string(
            X_A,
            precision=12
        ) + "\n\n"
    )

    file.write(
        "PROTOCOL B FINAL STATE\n"
    )

    file.write("-" * 75 + "\n")

    file.write(
        np.array2string(
            X_B,
            precision=12
        ) + "\n\n"
    )

    file.write(
        "ANALYTICAL REFERENCE\n"
    )

    file.write("-" * 75 + "\n")

    file.write(
        f"Protocol A: {ANALYTICAL_A}\n"
    )

    file.write(
        f"Protocol B: {ANALYTICAL_B}\n\n"
    )

    file.write(
        "NUMERICAL DIFFERENCES: A - B\n"
    )

    file.write("-" * 75 + "\n")

    file.write(
        f"Dose difference: {DOSE_DIFFERENCE}\n"
    )

    file.write(
        f"Weighted-area difference: "
        f"{AREA_DIFFERENCE:.12e}\n"
    )

    file.write(
        f"Higher-order difference: "
        f"{HIGHER_ORDER_DIFFERENCE:.12f}\n\n"
    )

    file.write(
        "NUMERICAL ERRORS\n"
    )

    file.write("-" * 75 + "\n")

    file.write(
        f"Protocol A endpoint error: "
        f"{ENDPOINT_ERROR_A:.12e}\n"
    )

    file.write(
        f"Protocol B endpoint error: "
        f"{ENDPOINT_ERROR_B:.12e}\n"
    )

    file.write(
        f"Dose mismatch: {DOSE_ERROR:.12e}\n"
    )

    file.write(
        f"Area mismatch: {AREA_ERROR:.12e}\n"
    )

    file.write(
        f"Higher-order error: "
        f"{HIGHER_ORDER_ERROR:.12e}\n\n"
    )

    file.write(
        "LIE-BRACKET STRUCTURE\n"
    )

    file.write("-" * 75 + "\n")

    file.write(
        f"Second-order bracket: {BRACKET_SECOND}\n"
    )

    file.write(
        f"Nested third-order bracket: {BRACKET_THIRD}\n"
    )

    file.write(
        f"Second-order bracket norm: "
        f"{BRACKET_SECOND_NORM:.8f}\n"
    )

    file.write(
        f"Nested bracket norm: "
        f"{BRACKET_THIRD_NORM:.8f}\n\n"
    )

    file.write(
        "VALIDATION\n"
    )

    file.write("-" * 75 + "\n")

    file.write(
        "Equal-dose validation passed.\n"
    )

    file.write(
        "Equal-endpoint validation passed.\n"
    )

    file.write(
        "Equal-weighted-area validation passed.\n"
    )

    file.write(
        "Different higher-order response validated.\n"
    )

    file.write(
        "Analytical-numerical comparison passed.\n\n"
    )

    file.write(
        "INTERPRETATION\n"
    )

    file.write("-" * 75 + "\n")

    file.write(
        "This example establishes higher-order path dependence "
        "for the specified control-affine model.\n"
    )

    file.write(
        "It does not establish biological validation or "
        "a universal biological law.\n"
    )


# ============================================================
# 18. FINAL AUTOMATED VALIDATION
# ============================================================

checks = {
    "Protocol A analytical agreement":
        ENDPOINT_ERROR_A < 1e-10,

    "Protocol B analytical agreement":
        ENDPOINT_ERROR_B < 1e-10,

    "Equal total dose":
        DOSE_ERROR < 1e-10,

    "Equal weighted-area integral":
        AREA_ERROR < 1e-10,

    "Higher-order difference":
        HIGHER_ORDER_ERROR < 1e-10,

    "Finite numerical states":
        np.all(np.isfinite(X_A))
        and np.all(np.isfinite(X_B)),

    "Valid bracket calculations":
        np.isclose(BRACKET_SECOND_NORM, 1.0)
        and np.isclose(BRACKET_THIRD_NORM, 2.0)
}

for check_name, passed in checks.items():

    if not passed:

        raise RuntimeError(
            f"Validation failed: {check_name}"
        )


# ============================================================
# 19. PRINT FINAL RESULTS
# ============================================================

print("\nPHASE 4B COMPLETED")
print("=" * 75)

print("\nPROTOCOL A")
print("-" * 75)
print("Final state:", X_A)

print("\nPROTOCOL B")
print("-" * 75)
print("Final state:", X_B)

print("\nDOSE AND AREA COMPARISON")
print("-" * 75)

print(
    "Dose difference:",
    DOSE_DIFFERENCE
)

print(
    f"Weighted-area difference: "
    f"{AREA_DIFFERENCE:.12e}"
)

print("\nHIGHER-ORDER RESPONSE")
print("-" * 75)

print(
    f"Numerical difference: "
    f"{HIGHER_ORDER_DIFFERENCE:.12f}"
)

print(
    f"Analytical difference: "
    f"{ANALYTICAL_HIGHER_ORDER_DIFFERENCE:.12f}"
)

print(
    f"Absolute error: "
    f"{HIGHER_ORDER_ERROR:.12e}"
)

print("\nLIE-BRACKET STRUCTURE")
print("-" * 75)

print(
    "Second-order bracket:",
    BRACKET_SECOND
)

print(
    "Nested third-order bracket:",
    BRACKET_THIRD
)

print("\nVALIDATION")
print("-" * 75)

for check_name, passed in checks.items():

    status = "PASSED" if passed else "FAILED"

    print(f"{check_name}: {status}")

print("\nOUTPUT DIRECTORY")
print("-" * 75)

print(
    os.path.abspath(OUTPUT_DIR)
)

print("\nGENERATED FILES")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(" -", filename)

print("\nALL PHASE 4B CHECKS PASSED.")
