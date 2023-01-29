import dataclasses
import numpy as np
from typing import Union, Tuple, List
from scipy.constants import constants
import matplotlib.pyplot as plt

from pyfit.solver import TimeDomainSolver
from tasks import SignalGeneration
from tasks.BoxResonator import BoxWithElectrodes
from tasks.task5Database import Database, CompressedDatabase, DatabaseParent


def setup_leapfrog(box: BoxWithElectrodes, signal: SignalGeneration.Signal) -> Tuple[np.ndarray, np.ndarray, float]:
    t_step = TimeDomainSolver.cfl_condition(box.mesh, box.idx_primary_edges, constants.mu_0, constants.epsilon_0)
    return TimeDomainSolver.time_axis(signal.t_start, signal.t_end, t_step)


def run_leapfrog(box: BoxWithElectrodes, cube_position: np.ndarray, width_cube: float, id_electrode_current: int,
                 signal: Union[SignalGeneration.Signal, np.ndarray], t_axis: Tuple[np.ndarray, np.ndarray, float]) \
        -> Tuple[np.ndarray, np.ndarray]:

    if isinstance(signal, SignalGeneration.Signal):
        sig = signal.current
    else:
        sig = signal

    # electric boundary conditions on cube:
    idx_primary_edges = box.boundary_cube(*cube_position, width=width_cube)

    # get operators:
    reluctance_dg, capacitance_pg, curl_operator = box.operators(idx_primary_edges, box.idx_dual_edges)
    current_distribution = box.current_distribution_electrode(id_electrode_current, idx_primary_edges)

    # run the leapfrog algorithm and measure voltage at electrode b:
    _, v_a, v_b = TimeDomainSolver.leapfrog(t_axis, reluctance_dg, capacitance_pg, curl_operator, current_distribution,
                                            sig, box.idx_electrode_a_x, box.idx_electrode_b_x)
    return v_a, v_b


def plot_signal_over_time(t_axis: np.ndarray, signal: np.ndarray, cube_lfb: Tuple[float, float, float] = None):
    fig, axs = plt.subplots()
    axs.plot(t_axis * 1e9, signal)
    axs.set_xlabel("time in (ns)")
    axs.set_ylabel("voltage in (V)")
    axs.grid()
    if cube_lfb is not None:
        cube_lfb = np.round(cube_lfb, 2)
        axs.set_title(f"Left front bottom of cube at {cube_lfb}")
    return fig, axs


def setup_compressed_database(db: Database, compression: float) -> CompressedDatabase:
    with CompressedDatabase(db.properties, compression=compression) as dbc:
        if not dbc.created:
            dbc.create_database(*db.get_entries())
    return dbc


def measurement_signal(box: BoxWithElectrodes, signal: SignalGeneration.Signal, displacement: Union[float, np.ndarray],
                       width_cube: float):
    cube_lfb = np.array([np.min(box.mesh.x_coord), np.min(box.mesh.y_coord), np.min(box.mesh.z_coord)]) + displacement

    t_axis = setup_leapfrog(box, signal)

    # forward step:
    _, v_b_fw = run_leapfrog(box, cube_lfb, width_cube, box.id_electrode_a, signal, t_axis)

    # time reversal:
    current_tr = SignalGeneration.time_reverse_signal(v_b_fw)
    v_a_tr, _ = run_leapfrog(box, cube_lfb, width_cube, box.id_electrode_b, current_tr, t_axis)

    # flip time reversed measured signal in order to be comparable:
    v_a_tr = np.flip(v_a_tr)

    # scale the time reversed measured signal to same scale of values as forward signal:
    v_a_tr *= np.max(v_b_fw) / np.max(v_a_tr)

    return v_b_fw, v_a_tr


def plot_likelihood(xyz: np.ndarray, score: np.ndarray, method: str = ""):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    color_map = plt.get_cmap("YlOrRd")
    xyz *= 1e3
    plot = ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=score, cmap=color_map)
    fig.colorbar(plot)
    ax.set_xlabel("x-pos (mm)")
    ax.set_ylabel("y-pos (mm)")
    ax.set_zlabel("z-pos (mm)")
    fig.suptitle("Likelihood of position of cube's lower left front corner", fontweight="semibold")
    ax.set_title(method + f" -- most likely: ({np.round(xyz[0, 0]).astype(int)}, {np.round(xyz[0, 1]).astype(int)},"
                 f" {np.round(xyz[0, 2]).astype(int)})mm")
    return fig, ax


def search_databases(signal: np.ndarray, full_database: Database, compressed_database: CompressedDatabase,
                     label: str, plot_flag: bool = True, print_flag: bool = True):
    res_full = full_database.search(signal)
    res_compressed = compressed_database.search(signal, method="correlation")
    res_restored = compressed_database.search(signal, method="restore-correlation")
    if plot_flag:
        plot_likelihood(*res_full, method=label + "full db")
        plot_likelihood(*res_restored, method=label + "restored db")
        plot_likelihood(*res_compressed, method=label + "compressed db")
    if print_flag:
        def print_result(result: Tuple[np.ndarray, np.ndarray], title: str):
            xyz, score = result
            xyz = np.round(xyz*1e3).astype(int)
            print(f"--> {title}")
            print(f"   x (mm)      y (mm)      z (mm)     score")
            for k in range(3):
                print(f"   {xyz[k][0]}     {xyz[k][1]}     {xyz[k][2]}     {score[k]}")
            print("\n")
        print(f"Results of {label}:")
        print(f"-------------------")
        print_result(res_full, "full_database")
        print_result(res_compressed, "compressed_database")
        print_result(res_restored, "compressed_restored_database")

    return res_full, res_compressed, res_restored


def search_single_database(signal: np.ndarray, database: DatabaseParent, label: str, plot_flag: bool = True):
    result = database.search(signal)
    if plot_flag:
        plot_likelihood(*result, method=label)
    return result


@dataclasses.dataclass
class Study:
    label: str = "task5"
    sides_box: np.ndarray = np.array([0.200, 0.160, 0.120])
    sides_electrode: np.ndarray = np.array([0.04, 0.08, 0])  # [m] : sizes of a electrode
    id_electrode_a: int = 11
    id_electrode_b: int = 12
    mesh_size_max: np.ndarray = np.array([.02, .02, .02])  # [m] : maximal mesh sizes in x, y, z direction
    signal_current_amplitude: float = 1e-6  # [A]
    signal_freq: float = 5e9  # [Hz]
    signal_t_start: float = 0  # [s] starting time
    signal_t_end: float = 20e-9  # [s] end time
    signal_gauss_standard_deviation: float = 0.5e-9  # [s] standard deviation for applied gaussian function
    signal_t_gauss = 1e-9  # [s] time when gaussian function reaches its maximal value
    width_cube = 0.02  # [m] : width of the cubes sizes
    box: BoxWithElectrodes = None
    signal: SignalGeneration.Signal = None
    full_database_a: Database = None
    full_database_b: Database = None
    compressed_database_a: CompressedDatabase = None
    compressed_database_b: CompressedDatabase = None
    compression_fw = 20 / 480  # trial and error -> first 20 of 480 singularities needed
    compression_tr = 300 / 480

    def __enter__(self):
        self.create_model()
        self.__setup_full_database()
        self.__setup_compressed_database()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def create_model(self):
        self.box = BoxWithElectrodes(self.sides_box, self.sides_electrode, self.id_electrode_a, self.id_electrode_b)
        self.box.set_mesh(self.mesh_size_max)
        self.signal = SignalGeneration.Signal(self.signal_current_amplitude, self.signal_freq, self.signal_freq,
                                              self.signal_t_start, self.signal_t_end,
                                              self.signal_gauss_standard_deviation, self.signal_t_gauss)

    def __all_cube_positions(self) -> np.ndarray:
        # positions of the lower left corner of cube
        x, y, z = self.box.mesh.x_coord[:-1], self.box.mesh.y_coord[:-1], self.box.mesh.z_coord[:-1]
        num_combinations = x.shape[0] * y.shape[0] * z.shape[0]
        combinations = np.zeros((num_combinations, 3))
        i = 0
        for x_i in x:
            for y_i in y:
                for z_i in z:
                    combinations[i, :] = np.array([x_i, y_i, z_i])
                    i += 1
        return combinations

    def __setup_full_database(self):

        # setup for time domain solver:
        t_axis = setup_leapfrog(self.box, self.signal)

        with Database(self.box, self.signal, self.width_cube, "A") as db_a,\
                Database(self.box, self.signal, self.width_cube, "B") as db_b:
            databases = (db_a, db_b)
            # save time axis to database:
            for db in databases:
                if not db.has_time_axis:
                    db.set_time_axis(t_axis[1])  # dual time axis

            # add entries for voltage measured at electrode a, b:
            cube_positions = self.__all_cube_positions()
            for k in np.arange(cube_positions.shape[0]):
                if all([db.get_entry_id(*cube_positions[k, :]) != -1 for db in databases]):
                    continue

                voltage_a, voltage_b = run_leapfrog(self.box, cube_positions[k, :], self.width_cube,
                                                    self.box.id_electrode_a, self.signal, t_axis)

                # save entry to database:
                db_a.add_entry(*cube_positions[k, :], measured_voltage=voltage_a)
                db_b.add_entry(*cube_positions[k, :], measured_voltage=voltage_b)

        self.full_database_a = db_a
        self.full_database_b = db_b

    def __setup_compressed_database(self):
        def compress_database(db: Database, compression: float) -> CompressedDatabase:
            with CompressedDatabase(db.properties, compression=compression) as dbc:
                if not dbc.created:
                    dbc.create_database(*db.get_entries())
            return dbc
        self.compressed_database_a = compress_database(self.full_database_a, self.compression_tr)
        self.compressed_database_b = compress_database(self.full_database_b, self.compression_fw)

    def run(self, measurement_signal_b: np.ndarray = None, measurement_signal_a: np.ndarray = None,
            compression: bool = True, label: str = None):

        results = {}

        if compression:
            if measurement_signal_b is not None:
                res_b = search_databases(measurement_signal_b, self.full_database_b, self.compressed_database_b,
                                         self.label + " fw ")
                results["b_full"], results["b_compressed"], results["b_restored"] = res_b

            if measurement_signal_a is not None:
                res_a = search_databases(measurement_signal_a, self.full_database_a, self.compressed_database_a,
                                         self.label + " tr ")
                results["a_full"], results["a_compressed"], results["a_restored"] = res_a
        else:
            if measurement_signal_b is not None:
                results["b_"+label] = search_single_database(measurement_signal_b, self.full_database_b, label + " fw ")
            if measurement_signal_a is not None:
                results["a_" + label] = search_single_database(measurement_signal_a, self.full_database_a,
                                                               label + " tr ")

        return results

    def measurement_signal(self, displacement: Union[float, np.ndarray]):
        cube_lfb = np.array(
            [np.min(self.box.mesh.x_coord), np.min(self.box.mesh.y_coord), np.min(self.box.mesh.z_coord)]) \
                   + displacement

        t_axis = setup_leapfrog(self.box, self.signal)

        # forward step:
        _, v_b_fw = run_leapfrog(self.box, cube_lfb, self.width_cube, self.box.id_electrode_a, self.signal, t_axis)

        # time reversal:
        current_tr = SignalGeneration.time_reverse_signal(v_b_fw)
        v_a_tr, _ = run_leapfrog(self.box, cube_lfb, self.width_cube, self.box.id_electrode_b, current_tr, t_axis)

        # flip time reversed measured signal in order to be comparable:
        v_a_tr = np.flip(v_a_tr)

        # scale the time reversed measured signal to same scale of values as forward signal:
        v_a_tr *= np.max(v_b_fw) / np.max(v_a_tr)

        return v_b_fw, v_a_tr


def parameter_sweep(displacement: float, **kwargs):
    results = {}

    for key, values in kwargs.items():
        for value in values:
            label = key + str(value)
            with Study(**{key: value}) as study:
                signal = study.measurement_signal(displacement)
                results[label] = study.run(*signal, compression=False, label=label)
    return results


def main():
    displacement = 40e-3  # [m] displacement in the x, y, z direction of left front bottom point of cube and box

    with Study(label="task5") as study_task5:
        measurement_signal_task5 = study_task5.measurement_signal(displacement)  # (forward: B and time reversed: A)
        results_task5 = study_task5.run(*measurement_signal_task5)

    # with Study(label="smaller electrodes", sides_electrode=np.array([0.04, 0.08, 0])/2) as study_smaller_electrodes:
    #     measurement_signal_smaller_electrodes = study_smaller_electrodes.measurement_signal(displacement)
    #     results_smaller_electrodes = study_smaller_electrodes.run(*measurement_signal_smaller_electrodes)

    with Study(label="greater t_end", signal_t_end=45e-9) as study_t_end:
        measurement_signal_t_end = study_t_end.measurement_signal(displacement)
        results_t_end = study_t_end.run(*measurement_signal_t_end)

    # results_freq = parameter_sweep(displacement, signal_freq=[2e9, 3e9, 4e9, 5e9, 6e9, 7e9, 8e9])
    # results_t_end = parameter_sweep(displacement, signal_t_end=[20e-9, 30e-9, 35e-9, 40e-9, 45e-9])
    # results_sigma = parameter_sweep(displacement, signal_gauss_standard_deviation=[.25e-9, .5e-9, 1e-9])
    # results_electrodes = parameter_sweep(displacement, sides_electrode=[np.array([0.06, 0.1, 0]),
    #                                                                     np.array([0.04, 0.08, 0]),
    #                                                                     np.array([0.02, 0.04, 0]),
    #                                                                     np.array([0.01, 0.02, 0])])

    plt.show()


if __name__ == "__main__":
    main()

