import numpy as np
import matplotlib.pyplot as plt
from pyfit.solver import TimeDomainSolver
from BoxResonator import BoxWithElectrodes
import SignalGeneration


def plot_current(t_axis: np.ndarray, i: np.ndarray, title: str = None):
    """Plot the current (used in the forward step)."""
    fig, axs = plt.subplots()
    if title is not None:
        fig.suptitle(title, fontsize="x-large", fontweight="bold")
    axs.plot(t_axis * 1e9, i * 1e6)
    axs.set_xlabel("time (ns)")
    axs.set_ylabel("current (µA)")
    axs.grid()
    return fig, axs


def plot_electrodes(t_axis: np.ndarray, v_electrode_a: np.ndarray, v_electrode_b: np.ndarray, title: str = None):
    """Plot the voltage on measured on both electrodes."""
    fig, axs = plt.subplots(2, 1, sharex="col")
    if title is not None:
        fig.suptitle(title, fontsize="x-large", fontweight="bold")
    axs[0].plot(t_axis * 1e9, v_electrode_a / np.max(v_electrode_a))
    axs[1].plot(t_axis * 1e9, v_electrode_b / np.max(v_electrode_b))
    axs[1].set_xlabel("time (ns)")
    axs[0].set_title("Electrode A")
    axs[1].set_title("Electrode B")
    for ax in axs:
        ax.grid()
        ax.set_ylabel("voltage (normalised)")
    return fig, axs


def plot_compare(t_axis: np.ndarray, v_forward: np.ndarray, v_time_reversal: np.ndarray, title: str = None):
    """Compare voltage measured during forward step with the time reversal step."""
    fig, axs = plt.subplots()
    if title is not None:
        fig.suptitle(title, fontsize="x-large", fontweight="bold")
    axs.plot(t_axis * 1e9, v_forward / np.max(v_forward), label="forward")
    axs.plot(t_axis * 1e9, np.flip(v_time_reversal / np.max(v_time_reversal)), label="reversed tr")
    axs.set_xlabel("time in (ns)")
    axs.set_ylabel("voltage (normalised)")
    axs.grid()
    axs.legend()
    return fig, axs


def main():
    # data:
    sides_box = np.array([0.200, 0.160, 0.120])  # [m] : domain sizes (x, y, z)
    mesh_size_max = np.array([.04, .04, .04]) / 4  # [m] : maximal mesh sizes in x, y, z direction
    electrodes = {"sides_electrode": np.array([0.04, 0.08, 0]),  # [m] : sizes of a electrode
                  "id_electrode_a": 11, "id_electrode_b": 12}
    signal_properties = {"current_amplitude": 1e-6,  # [A]
                         "f_0": 5e9,  # [Hz]
                         "f_1": 5e9,  # [Hz]
                         "t_start": 0,  # [s] starting time
                         "t_end": 20e-9,  # [s] end time
                         "gauss_standard_deviation": 0.5e-9,  # [s] standard deviation for applied gaussian function
                         "t_gauss": 1e-9  # [s] time when gaussian function reaches its maximal value
                         }

    # create model:
    box = BoxWithElectrodes(sides_box, **electrodes)
    box.set_mesh(mesh_size_max)
    reluctance_dg, capacitance_pg, curl_operator = box.operators(box.idx_primary_edges, box.idx_dual_edges)

    # ---------- forward step: ----------
    # define current:
    signal = SignalGeneration.Signal(**signal_properties)
    current_distribution_fw = box.current_distribution_electrode(box.id_electrode_a)

    t_axis = box.get_t_axis_leapfrog(signal)  # (t_axis_primary, t_axis_dual, t_step)

    current_fw, voltage_a_fw, voltage_b_fw = TimeDomainSolver.leapfrog(t_axis, reluctance_dg, capacitance_pg,
                                                                       curl_operator, current_distribution_fw,
                                                                       signal.current, box.idx_electrode_a_x,
                                                                       box.idx_electrode_b_x)

    # ---------- time reversal: ----------
    current_distribution_tr = box.current_distribution_electrode(box.id_electrode_b)
    current_tr = SignalGeneration.time_reverse_signal(voltage_b_fw)  # a resistance of 1 Ohm is assumed
    _, voltage_a_tr, voltage_b_tr = TimeDomainSolver.leapfrog(t_axis, reluctance_dg, capacitance_pg, curl_operator,
                                                              current_distribution_tr, current_tr,
                                                              box.idx_electrode_a_x, box.idx_electrode_b_x)

    # plotting:
    plot_current(t_axis[0], current_fw, title="forward step: current")
    plot_electrodes(t_axis[1], voltage_a_fw, voltage_b_fw, title="forward step")
    plot_electrodes(t_axis[1], voltage_a_tr, voltage_b_tr, title="time reversal")
    plot_compare(t_axis[1], voltage_a_fw, voltage_a_tr, title="Electrode A")
    plt.show()


if __name__ == "__main__":
    main()

