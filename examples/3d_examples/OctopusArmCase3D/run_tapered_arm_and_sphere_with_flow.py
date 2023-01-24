import numpy as np
from set_environment_tapered_arm_and_sphere_with_flow import Environment
from arm_functions_3d import SigmoidActivationLongitudinalMuscles  # , LocalActivation
import sopht.simulator as sps
import sopht.utils as spu
import elastica as ea
import click
from matplotlib import pyplot as plt
# from oscillation_activation_functions import OscillationActivation


def tapered_arm_and_cylinder_flow_coupling(
    non_dimensional_final_time: float,
    n_elems: int,
    slenderness_ratio: float,
    cauchy_number: float,
    mass_ratio: float,
    reynolds_number: float,
    taper_ratio: float,
    activation_period:float,
    activation_level_max:float,
    grid_size: tuple[int, int, int],
    surface_grid_density_for_largest_element : int,
    coupling_stiffness: float = -2e2,
    coupling_damping: float = -1e-1,
    num_threads: int = 4,
    precision: str = "single",
    save_data: bool = True,
) -> None:
    # =================COMMON STUFF BEGIN=====================
    grid_dim = 3
    grid_size_z, grid_size_y, grid_size_x = grid_size
    real_t = spu.get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()
    period = activation_period
    rho_f = 1
    base_length = 1.0
    vel_scale = base_length / period
    final_time = period * non_dimensional_final_time
    # x_range = 2.5 * base_length
    # y_range = grid_size_y / grid_size_x * x_range
    # z_range = grid_size_z / grid_size_x * x_range
    y_range = 1.8 * base_length
    x_range = grid_size_x / grid_size_y * y_range
    z_range = grid_size_z / grid_size_y * y_range
    # 2x1 domain with x_range = 4 * base_length

    # =================PYELASTICA STUFF BEGIN=====================
    rod_dt = 3.0e-4
    env = Environment(final_time, time_step=rod_dt, rendering_fps=30)
    rho_s = mass_ratio * rho_f
    base_diameter = base_length / slenderness_ratio
    base_radius = base_diameter / 2
    base_area = np.pi * base_radius**2
    moment_of_inertia = np.pi / 4 * base_radius**4
    # Cau = (rho_f U^2 L^3 D) / EI
    youngs_modulus = (rho_f * vel_scale**2 * base_length**3 * base_diameter) / (
        cauchy_number * moment_of_inertia
    )
    # Es_Eb = stretch_bending_ratio * moment_of_inertia / (base_area * base_length**2)
    poisson_ratio = 0.5
    shear_modulus = youngs_modulus / (2 * (1 + poisson_ratio))

    start = np.zeros(grid_dim) + np.array(
        [0.3 * x_range, 0.5 * y_range, 0.12 * z_range]
    )
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])

    radius = np.linspace(base_radius, base_radius / taper_ratio, n_elems + 1)
    radius_mean = (radius[:-1] + radius[1:]) / 2

    shearable_rod = ea.CosseratRod.straight_rod(
        n_elements=n_elems,
        start=start,
        direction=direction,
        normal=normal,
        base_length=base_length,
        base_radius=radius_mean.copy(),
        density=rho_s,
        nu=0.0,  # internal damping constant, deprecated in v0.3.0
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )

    # Initialize fixed sphere (elastica rigid body)
    sphere_com = np.array([0.6667 * x_range, 0.5 * y_range, 0.3111 * z_range])
    sphere_diameter = 3 * radius_mean[0]
    sphere = ea.Sphere(
        center=sphere_com,
        base_radius=sphere_diameter/2,
        density=rho_s,
    )

    env.reset(youngs_modulus, rho_s, shearable_rod, sphere)

    # Setup activation functions to control muscles
    activations = []
    activation_functions = []
    for m in range(len(env.muscle_groups)):
        activations.append(np.zeros(env.muscle_groups[m].activation.shape))

        if m == 2:
            # activation_functions.append(
            #     OscillationActivation(
            #         wave_number=0.05,
            #         frequency=1 / activation_period,  # f_p,
            #         phase_shift=0,  # X_p,
            #         start_time=0.0,
            #         end_time=10000,
            #         start_non_dim_length=0,
            #         end_non_dim_length=1.0,
            #         n_elems=n_elems,
            #         activation_level_max=0.2,
            #         a=10,
            #         b=0.5,
            #     )
            # )
            activation_functions.append(
                SigmoidActivationLongitudinalMuscles(
                    beta=1,
                    tau=period,
                    start_time=1.0,
                    end_time=10,  # 0.1 + 2 + 0.1 * 10,
                    start_non_dim_length=0,
                    end_non_dim_length=1.0,
                    activation_level_max=activation_level_max,
                    activation_level_end=activation_level_max,
                    activation_lower_threshold=0.0,
                    n_elems=n_elems,
                )
            )
    #     activation_functions.append(
    #         LocalActivation(
    #             ramp_interval=1.0,
    #             ramp_up_time=0.0,
    #             ramp_down_time=15,
    #             start_idx=0,
    #             end_idx=n_elems,
    #             activation_level=0.1,
    #         )
    #     )
        else:
            activation_functions.append(None)


    # =================PYELASTICA STUFF END=====================
    # ==================FLOW SETUP START=========================
    # Flow parameters
    kinematic_viscosity = base_diameter * vel_scale / reynolds_number
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=kinematic_viscosity,
        flow_type="navier_stokes_with_forcing",
        with_free_stream_flow=False,
        real_t=real_t,
        num_threads=num_threads,
        navier_stokes_inertial_term_form="rotational",
        filter_vorticity=True,
        filter_setting_dict={"order": 3, "type": "convolution"},
    )
    # ==================FLOW SETUP END=========================
    # ==================FLOW-ROD COMMUNICATOR SETUP START======
    flow_body_interactors: list[
        sps.RigidBodyFlowInteraction | sps.CosseratRodFlowInteraction
    ] = []
    cosserat_rod_flow_interactor = sps.CosseratRodFlowInteraction(
        cosserat_rod=env.shearable_rod,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=grid_dim,
        real_t=real_t,
        num_threads=num_threads,
        forcing_grid_cls=sps.CosseratRodSurfaceForcingGrid,
        surface_grid_density_for_largest_element=surface_grid_density_for_largest_element
    )
    flow_body_interactors.append(cosserat_rod_flow_interactor)

    env.simulator.add_forcing_to(env.shearable_rod).using(
        sps.FlowForces,
        cosserat_rod_flow_interactor,
    )

    num_forcing_points_along_equator = int(
        1.875 * sphere_diameter / y_range * grid_size_y
    )
    sphere_flow_interactor = sps.RigidBodyFlowInteraction(
        rigid_body=sphere,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=grid_dim,
        real_t=real_t,
        forcing_grid_cls=sps.SphereForcingGrid,
        num_forcing_points_along_equator=num_forcing_points_along_equator,
    )
    flow_body_interactors.append(sphere_flow_interactor)
    env.simulator.add_forcing_to(sphere).using(
        sps.FlowForces,
        sphere_flow_interactor,
    )
    # ==================FLOW-ROD COMMUNICATOR SETUP END======

    if save_data:
        # setup IO
        # TODO internalise this in flow simulator as dump_fields
        io_origin = np.array(
            [
                flow_sim.position_field[z_axis_idx].min(),
                flow_sim.position_field[y_axis_idx].min(),
                flow_sim.position_field[x_axis_idx].min(),
            ]
        )
        io_dx = flow_sim.dx * np.ones(grid_dim)
        io_grid_size = np.array(grid_size)
        # Initialize flow eulerian grid IO
        io = spu.IO(dim=grid_dim, real_dtype=real_t)
        io.define_eulerian_grid(origin=io_origin, dx=io_dx, grid_size=io_grid_size)
        io.add_as_eulerian_fields_for_io(
            vorticity=flow_sim.vorticity_field, velocity=flow_sim.velocity_field
        )
        # Initialize rod io
        rod_io = spu.CosseratRodIO(
            cosserat_rod=shearable_rod, dim=grid_dim, real_dtype=real_t
        )
        # Initialize sphere io
        sphere_io = spu.IO(dim=grid_dim, real_dtype=real_t)
        # Add vector field on lagrangian grid
        sphere_io.add_as_lagrangian_fields_for_io(
            lagrangian_grid=sphere_flow_interactor.forcing_grid.position_field,
            lagrangian_grid_name="sphere",
            vector_3d=sphere_flow_interactor.lag_grid_forcing_field,
        )
    # =================TIMESTEPPING====================
    # Finalize the pyelastica environment
    _, _ = env.finalize()

    foto_timer = 0.0
    foto_timer_limit = period / 20
    time_history = []
    arm_force_history = []
    sphere_force_history = []

    # create fig for plotting flow fields
    fig, ax = spu.create_figure_and_axes()

    while flow_sim.time < final_time:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            if save_data:
                io.save(
                    h5_file_name="sopht_"
                    + str("%0.4d" % (flow_sim.time * 100))
                    + ".h5",
                    time=flow_sim.time,
                )
                rod_io.save(
                    h5_file_name="rod_" + str("%0.4d" % (flow_sim.time * 100)) + ".h5",
                    time=flow_sim.time,
                )
                sphere_io.save(
                    h5_file_name="sphere_"
                    + str("%0.4d" % (flow_sim.time * 100))
                    + ".h5",
                    time=flow_sim.time,
                )
                env.save_data()

            ax.set_title(f"Vorticity magnitude, time: {flow_sim.time / final_time:.2f}")
            contourf_obj = ax.contourf(
                flow_sim.position_field[x_axis_idx, :, grid_size_y // 2, :],
                flow_sim.position_field[z_axis_idx, :, grid_size_y // 2, :],
                # TODO have a function for computing velocity magnitude
                np.linalg.norm(
                    np.mean(
                        flow_sim.velocity_field[
                            :, :, grid_size_y // 2 - 1 : grid_size_y // 2 + 1, :
                        ],
                        axis=2,
                    ),
                    axis=0,
                ),
                levels=np.linspace(0, vel_scale, 50),
                extend="both",
                cmap="Purples",
            )
            cbar = fig.colorbar(mappable=contourf_obj, ax=ax)
            ax.scatter(
                cosserat_rod_flow_interactor.forcing_grid.position_field[x_axis_idx],
                cosserat_rod_flow_interactor.forcing_grid.position_field[z_axis_idx],
                s=5,
                color="k",
            )
            ax.scatter(
                sphere_flow_interactor.forcing_grid.position_field[x_axis_idx],
                sphere_flow_interactor.forcing_grid.position_field[z_axis_idx],
                s=5,
                color="k",
            )
            spu.save_and_clear_fig(
                fig,
                ax,
                cbar,
                file_name="snap_" + str("%0.5d" % (flow_sim.time * 100)) + ".png",
            )

            plt.rcParams.update({"font.size": 22})
            fig_2 = plt.figure(figsize=(10, 10), frameon=True, dpi=150)
            axs = []
            axs.append(plt.subplot2grid((1, 1), (0, 0)))
            axs[0].plot(
                env.shearable_rod.velocity_collection[x_axis_idx],
            )
            axs[0].plot(
                env.shearable_rod.velocity_collection[y_axis_idx],
            )
            axs[0].plot(
                env.shearable_rod.velocity_collection[z_axis_idx],
            )
            axs[0].set_xlabel("idx", fontsize=20)
            axs[0].set_ylabel("vel", fontsize=20)
            axs[0].set_ylim(-1.5, 1.5)
            plt.tight_layout()
            fig_2.align_ylabels()
            fig_2.savefig("vel_" + str("%0.5d" % (flow_sim.time * 100)) + ".png")
            plt.close(plt.gcf())

            time_history.append(flow_sim.time)
            grid_dev_error = 0.0
            for flow_body_interactor in flow_body_interactors:
                grid_dev_error += (
                    flow_body_interactor.get_grid_deviation_error_l2_norm()
                )

            forces = np.sum(cosserat_rod_flow_interactor.lag_grid_forcing_field, axis=1)
            arm_force_history.append(forces.copy())
            forces = np.sum(sphere_flow_interactor.lag_grid_forcing_field, axis=1)
            sphere_force_history.append(forces.copy())

            print(
                f"time: {flow_sim.time:.2f} ({(flow_sim.time/final_time*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                f"vort divg. L2 norm: {flow_sim.get_vorticity_divergence_l2_norm():.4f}, "
                f"grid deviation L2 error: {grid_dev_error:.6f}"
            )

        # compute timestep
        flow_dt = flow_sim.compute_stable_timestep(dt_prefac=0.2)

        # timestep the rod, through the flow timestep
        rod_time_steps = int(flow_dt / min(flow_dt, rod_dt))
        local_rod_dt = flow_dt / rod_time_steps
        rod_time = flow_sim.time
        for i in range(rod_time_steps):
            # Activate longitudinal muscle
            activation_functions[2].apply_activation(
                shearable_rod, activations[2], rod_time
            )
            # Do one elastica step
            env.time_step = local_rod_dt
            rod_time, systems, done = env.step(rod_time, activations)

            # timestep the body_flow_interactors
            for flow_body_interactor in flow_body_interactors:
                flow_body_interactor.time_step(dt=local_rod_dt)

        # evaluate feedback/interaction between flow and bodies
        for flow_body_interactor in flow_body_interactors:
            flow_body_interactor()

        # timestep the flow
        flow_sim.time_step(dt=flow_dt)

        # update simulation time
        foto_timer += flow_dt

    # compile video
    spu.make_video_from_image_series(
        video_name="flow", image_series_name="snap", frame_rate=10
    )
    spu.make_video_from_image_series(
        video_name="rod_vel", image_series_name="vel", frame_rate=10
    )
    env.save_data()

    np.savetxt(
        "flow_forces_vs_time.csv",
        np.c_[
            np.array(time_history),
            np.array(arm_force_history),
            np.linalg.norm(np.array(arm_force_history), axis=1),
            np.array(sphere_force_history),
            np.linalg.norm(np.array(sphere_force_history), axis=1),
        ],
        delimiter=",",
        header="time, arm force x, arm force y, arm force z, arm force norm, sphere force x, sphere force y, sphere force z, sphere force norm",
    )

if __name__ == "__main__":

    @click.command()
    @click.option("--num_threads", default=4, help="Number of threads for parallelism.")
    @click.option("--nz", default=150, help="Number of grid points in z direction.")
    @click.option("--taper_ratio", default=7, help="Arm taper ratio.")
    @click.option("--activation_mag", default=0.15, help="Muscle activation magnitude.")
    @click.option("--period", default=0.5, help="Activation period.")
    def simulate_parallelised_octopus_arm(
        num_threads, nz, taper_ratio, activation_mag, period
    ):

        nx = nz
        ny = nz
        grid_size = (nz, ny, nx)
        surface_grid_density_for_largest_element = nz // 10
        exp_n_elem = nz // 3

        click.echo(f"Number of threads for parallelism: {num_threads}")
        final_time = 10

        exp_rho_s = 1044  # kg/m3
        exp_rho_f = 1022  # kg/m3
        exp_youngs_modulus = 1e4  # Pa
        exp_base_length = 0.2  # m
        exp_base_diameter = exp_base_length / 10  # m
        exp_kinematic_viscosity = 1e-6  # m2/s
        exp_activation_period = period  # 2.533 * 3
        exp_activation_level_max = activation_mag  # 0.2
        exp_U_free_stream = exp_base_length / exp_activation_period  # m/s
        exp_mass_ratio = exp_rho_s / exp_rho_f
        exp_slenderness_ratio = exp_base_length / exp_base_diameter
        exp_base_radius = exp_base_diameter / 2
        exp_base_area = np.pi * exp_base_radius**2
        exp_moment_of_inertia = np.pi / 4 * exp_base_radius**4
        exp_bending_rigidity = exp_youngs_modulus * exp_moment_of_inertia
        exp_cauchy_number = (
            exp_rho_f
            * exp_U_free_stream**2
            * exp_base_length**3
            * exp_base_diameter
            / exp_bending_rigidity
        )
        exp_Re = exp_U_free_stream * exp_base_diameter / exp_kinematic_viscosity
        # stretch to bending ratio EAL2 / EI
        exp_Ks_Kb = (exp_youngs_modulus * exp_base_area * exp_base_length**2) / (
            exp_youngs_modulus * exp_moment_of_inertia
        )
        exp_non_dimensional_final_time = final_time / period
        exp_taper_ratio = taper_ratio
        print(f"Re: {exp_Re}, Ca: {exp_cauchy_number}, Ks_Kb: {exp_Ks_Kb}")
        tapered_arm_and_cylinder_flow_coupling(
            non_dimensional_final_time=exp_non_dimensional_final_time,
            n_elems=exp_n_elem,
            slenderness_ratio=exp_slenderness_ratio,
            cauchy_number=exp_cauchy_number,
            mass_ratio=exp_mass_ratio,
            reynolds_number=exp_Re,
            taper_ratio=exp_taper_ratio,
            activation_period = exp_activation_period,
            activation_level_max = exp_activation_level_max,
            surface_grid_density_for_largest_element=surface_grid_density_for_largest_element,
            grid_size=grid_size,
            num_threads=num_threads,
            save_data=True,
        )

    simulate_parallelised_octopus_arm()
