import numpy as np
import matplotlib.pyplot as plt

# Constants
gamma = 1.4       # Heat capacity ratio (air)
R = 287           # Specific gas constant (J/kg·K)
L = 4.0           # Tube length (m)
dx = 0.05         # Step size (80 sections for 4m; more accuracy)
P0 = 2000         # Fixed initial pressure at x=0

# Temperature profiles
def get_temperature_profile(index):
    # Base temperature and gradient
    T0_values = [300, 500, 700, 900, 1100]
    gradients = [-0, -50, -100, -150, -200]

    T0 = T0_values[index]
    m = gradients[index]

    def T_profile(x):
        return T0 + m * x

    return T_profile, f"T(x) = {T0} {m:+}x"

# Pressure peaks
def count_peaks(pressure):
    peaks = []
    for i in range(1, len(pressure)-1):
        if pressure[i] > pressure[i-1] and pressure[i] > pressure[i+1] and pressure[i] > 0.5 * P0:
            peaks.append(i)
    return len(peaks), peaks

def solve_mode(target_mode, T_profile):
    # Discretize the tube
    x = np.arange(0, L + dx, dx)
    num_steps = len(x) - 1

    # Precompute temperature at each point
    T = np.array([T_profile(pos) for pos in x])
    c = np.sqrt(gamma * R * T)

    # Initial guess for omega based on average speed of sound
    c_avg = np.mean(c)

    # Special case for n=0 (DC mode)
    if target_mode == 0:
        # For n=0, we expect a very low frequency
        omega_guess = 0.1 * np.pi * c_avg / L
        omega_range = np.linspace(0.01 * omega_guess, 10 * omega_guess, 300)
    else:
        # For n>0, use normal estimation
        omega_guess = target_mode * np.pi * c_avg / L
        omega_range = np.linspace(0.5 * omega_guess, 2.0 * omega_guess, 300)

    # Store results for each omega
    results = []

    # ODE system
    def odes(x_idx, y, omega):
        P, Z = y
        current_T = T[x_idx]
        # Calculate temperature gradient
        if x_idx < num_steps:
            dTdx = (T[x_idx + 1] - T[x_idx]) / dx
        else:
            dTdx = (T[x_idx] - T[x_idx - 1]) / dx

        current_c = np.sqrt(gamma * R * current_T)
        dZdx = -(dTdx / current_T) * Z - (omega**2) / (current_c**2) * P
        dPdx = Z
        return np.array([dPdx, dZdx])

    # Iterate over omega values
    for omega in omega_range:
        # Initialize variables
        y = np.array([P0, 0.0])  # P(0)=2000, Z(0)=0
        P_sol = [y[0]]
        Z_sol = [y[1]]  # Store Z values for velocity calculation

        # Integrate using RK4
        for step in range(num_steps):
            k1 = odes(step, y, omega)
            k2 = odes(step, y + dx/2 * k1, omega)
            k3 = odes(step, y + dx/2 * k2, omega)
            k4 = odes(step, y + dx * k3, omega)
            y += dx/6 * (k1 + 2*k2 + 2*k3 + k4)
            P_sol.append(y[0])
            Z_sol.append(y[1])

        # Check both boundary condition at x=4 and the number of peaks
        P4 = y[0]
        abs_P = np.abs(P_sol)
        num_peaks, peak_indices = count_peaks(abs_P)

        # Store both P4 error and mode information
        results.append((omega, P4, P_sol, Z_sol, num_peaks, peak_indices))

    # Special case for n=0 (look for solutions with no peaks)
    if target_mode == 0:
        mode_solutions = [r for r in results if r[3] == 0]
    else:
        # Find solutions with exactly target_mode peaks
        mode_solutions = [r for r in results if r[4] == target_mode]

    if not mode_solutions:
        # Fall back to closest boundary condition
        best_idx = np.argmin(np.abs(np.array([res[1] for res in results])))
        best_omega, best_P4, best_P, best_Z, num_peaks, peak_indices = results[best_idx]
    else:
        # From the correct mode solutions, find the one with best boundary match
        best_idx = np.argmin(np.abs(np.array([res[1] for res in mode_solutions])))
        best_omega, best_P4, best_P, best_Z, num_peaks, peak_indices = mode_solutions[best_idx]

    return x, best_P, best_Z, best_omega, best_P4, num_peaks

# Process all five modes
for target_mode in [0, 1, 2, 3, 4]:
    # Store results to compare across temperature profiles
    frequency_results = []

    # Plot pressure distributions for each temperature profile
    plt.figure(figsize=(12, 8))

    for i in range(5):
        T_profile, label = get_temperature_profile(i)

        x, P, Z, omega, P4, num_peaks = solve_mode(target_mode, T_profile)

        # Take the modulus (absolute value) of pressure amplitudes
        P_abs = np.abs(P)

        plt.plot(x, P_abs, label=f'{label}, $f$={omega/(2*np.pi):.2f} Hz')
        frequency_results.append((label, omega, num_peaks))

    plt.xlabel('Distance (m)')
    plt.ylabel('Acoustic Pressure Amplitude ($Pa$)')
    plt.axvline(x=4, color='gray', linestyle='--', label='Open End')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'mode_{target_mode}_comparison_all_profiles.png')
    plt.show()

    # Print frequency results for this mode
    print(f"\nFrequency results for mode n={target_mode}:")
    print("=" * 60)
    print(f"{'Temperature Profile':<20} {'Frequency (Hz)':<15} {'Number of Peaks':<15}")
    print("-" * 60)
    for label, omega, num_peaks in frequency_results:
        print(f"{label:<20} {omega/(2*np.pi):<15.2f} {num_peaks:<15}")
    print("=" * 60)

    # Plot velocity amplitude distributions
    plt.figure(figsize=(12, 8))

    for i in range(5):
        T_profile, label = get_temperature_profile(i)

        x, P, Z, omega, P4, num_peaks = solve_mode(target_mode, T_profile)

        # Calculate temperature at each point
        T = np.array([T_profile(pos) for pos in x])

        # Calculate density at each point (from ideal gas law)
        P_mean = 101325  # Mean pressure (Pa) - atmospheric pressure
        rho = P_mean / (R * T)  # Density from ideal gas law

        # U' = abs(-1/(omega*rho) * dP/dx)
        # Since Z = dP/dx from your ODE system, we can use Z directly
        U_prime = np.abs(-1 / (omega * rho) * np.array(Z))

        plt.plot(x, U_prime, label=f'{label}, $f$={omega/(2*np.pi):.2f} Hz')

    plt.xlabel('Distance (m)')
    plt.ylabel('Acoustic Velocity Amplitude (m/s)')
    plt.axvline(x=4, color='gray', linestyle='--', label='Open End')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'mode_{target_mode}_velocity_amplitude.png')
    plt.show()