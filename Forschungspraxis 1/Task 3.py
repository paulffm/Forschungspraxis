import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
from typing import Tuple
from pyfit.solver import TimeDomainSolver

from tasks.task1 import AnalyticalSolution
import tasks.CurrentDistributions as current_distributions
from tasks.SignalGeneration import chirp, gaussian_curve

from tasks.BoxResonator import BoxResonator
from tasks import SignalGeneration


def fft(time_axis: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the fft for a given set of values.

    Parameters
    ----------
    time_axis : np.ndarray
        Time axis.
    values : np.ndarray
        Values of each element of the time axis.

    Returns
    -------
    frequency : np.ndarray
        Vector with frequency.
    values_fft : np.ndarray
        Vector with a value for each frequency.
    """
    if time_axis.shape != time_axis.shape:
        raise TypeError("Shape of time axis and value array must match!")
    num_sampling_points = values.shape[0]
    values_fft = scipy.fft.fft(values)
    values_fft = values_fft[:num_sampling_points//2]  # only positive part of spectrum is considered
    values_fft *= 2 / num_sampling_points  # see definition fft (scipy)
    frequency = scipy.fft.fftfreq(num_sampling_points, time_axis[1] - time_axis[0])[:num_sampling_points//2]
    return frequency, values_fft


def analytical_eigenfreq(w_x: float, w_y: float, w_z: float) -> np.ndarray:
    """
    Compute the eigenfrequencies analytically.

    Parameters
    ----------
    w_x : float
        Width of the box in x-direction [m].
    w_y : float
        Width of the box in y-direction [m].
    w_z : float
        Width of the box in z-direction [m].

    Returns
    -------
    ana_eig_freq : np.ndarray
        Eigenfrequencies that are greater that 1e9 GHz.
    """
    analytical_solution = AnalyticalSolution(w_x, w_y, w_z)
    ana_eig_freq = np.zeros(64)
    i = 0
    for n_x in np.arange(4):
        for n_y in np.arange(4):
            for n_z in np.arange(4):
                ana_eig_freq[i] = analytical_solution.eigenfrequencies(n_x, n_y, n_z)
                i += 1
    ana_eig_freq = np.sort(ana_eig_freq[ana_eig_freq > 1e9])
    return ana_eig_freq


def plot_fft(frequency: np.ndarray, values_fft: np.ndarray, f_min: float = 1e9, f_max: float = 5e9,
             ana_eig_freq: np.ndarray = None, y_label: str = None, title: str = None):
    """Plot the spectrum of the measured voltage."""
    fig, axs = plt.subplots()
    axs.plot(frequency * 1e-9, np.abs(values_fft))
    axs.set_xlim(f_min * 1e-9, f_max * 1e-9)
    axs.set_xlabel("frequency in (GHz)")
    if y_label is not None:
        axs.set_ylabel(y_label)
    if title is not None:
        axs.set_title(title)
    axs.grid()

    if ana_eig_freq is not None:
        axs.vlines(ana_eig_freq * 1e-9, ymin=0, ymax=np.max(np.abs(values_fft)) * 1.2, label="analytic", colors="C3",
                   linestyles="--")

    return fig, axs


def plot_measured_voltage(time_axis_primary: np.ndarray, time_axis_dual: np.ndarray, current_vals: np.ndarray,
                          voltage: np.ndarray, f0: float, f1: float):
    """Plot the applied current and measured voltage."""
    fig, axs = plt.subplots(2, 1, sharex="col")
    axs[0].plot(time_axis_primary * 1e9, current_vals * 1e6)
    axs[0].set_ylabel("current amplitude in (uA)")
    axs[0].grid()

    t0, t1 = time_axis_primary[0], time_axis_primary[-1]

    def time2freq(t):
        t *= 1e-9  # due to scaling of x-axis
        return f0 + (f1 - f0) * (t - t0) / (t1 - t0)

    def freq2time(f):
        return ((t1 - t0) * (f - f0) / (f1 - f0) + t0) * 1e9

    sec_ax = axs[0].secondary_xaxis("top", functions=(time2freq, freq2time))
    sec_ax.set_xlabel("frequency in (GHz)")
    axs[1].plot(time_axis_dual * 1e9, voltage * 1e3)
    axs[1].set_xlabel("time in (ns)")
    axs[1].set_ylabel("measured voltage in (mV)")
    axs[1].grid()

    return fig, axs


def plot_gauss(time_axis_primary: np.ndarray, variance_squared: float, amplitude: float, t_gaussian: float):
    """Plot the gaussian curve used to modulate the current."""
    fig, axs = plt.subplots(1, 1)
    axs.plot(time_axis_primary * 1e9, gaussian_curve(time_axis_primary, variance_squared, amplitude, t_gaussian))
    axs.set_xlabel("time in (ns)")
    axs.set_ylabel("Amplitude in (A)")
    axs.set_title("Gaussian curve")

    return fig, axs


def main():
    """Main function to run task3."""
    # data:
    sides_box = np.array([0.200, 0.160, 0.120])  # [m] : domain sizes (x, y, z)
    mesh_size_max = np.array([.04, .04, .04]) / 4  # [m] : maximal mesh sizes in x, y, z direction
    signal_properties = {"current_amplitude": 1e-6,  # [A]
                         "f_0": 1e9,  # [Hz]
                         "f_1": 5e9,  # [Hz]
                         "t_start": 0,  # [s] starting time
                         "t_end": 20e-9,  # [s] end time
                         "gauss_standard_deviation": 5e-9,  # [s] standard deviation for applied gaussian function
                         "t_gauss": 10e-9  # [s] time when gaussian function reaches its maximal value
                         }

    # create the model:
    box = BoxResonator(sides_box)
    box.set_mesh(mesh_size_max)
    signal = SignalGeneration.Signal(**signal_properties)

    # Determine a current distribution operator:
    current_distribution_operator = current_distributions.current_distribution_random(box.mesh, fill=0.25)

    # Build the operators needed for the leapfrog algorithm:
    current_distribution = current_distribution_operator[box.idx_primary_edges]
    reluctance_dg, capacitance_pg, curl_operator = box.get_operators()

    # Execute the leapfrog algorith and determine the voltage for each point on the dual axis:
    t_axis = box.get_t_axis_leapfrog(signal)  # (t_axis_primary, t_axis_dual, t_step)
    current, voltage, _ = TimeDomainSolver.leapfrog(t_axis, reluctance_dg, capacitance_pg, curl_operator,
                                                    current_distribution, signal.current)
    # Fouriertransform the measured voltage:
    freq, voltage_fft = fft(t_axis[1], voltage)

    # Plotting:
    plot_fft(freq, voltage_fft, f_min=signal_properties["f_0"], f_max=signal_properties["f_1"],
             y_label="abs voltage in (V)", ana_eig_freq=analytical_eigenfreq(*sides_box))
    plot_measured_voltage(t_axis[0], t_axis[1], current, voltage, signal_properties["f_0"], signal_properties["f_1"])
    plot_gauss(t_axis[0], signal_properties["gauss_standard_deviation"]**2, signal_properties["current_amplitude"],
               signal_properties["t_gauss"])
    plt.show()


if __name__ == "__main__":
    main()

