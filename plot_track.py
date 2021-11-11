import opt_mintime_traj
import numpy as np
import time
import json
import os
import trajectory_planning_helpers as tph
import copy
import matplotlib.pyplot as plt
import configparser
import pkg_resources
import helper_funcs_glob

"""
Created by:
Alexander Heilmeier

Documentation:
This script has to be executed to generate an optimal trajectory based on a given reference track.
"""

# ----------------------------------------------------------------------------------------------------------------------
# USER INPUT -----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# choose vehicle parameter file ----------------------------------------------------------------------------------------
file_paths = {"veh_params_file": "racecar.ini"}

# debug and plot options -----------------------------------------------------------------------------------------------
debug = True                                    # print console messages
plot_opts = {"mincurv_curv_lin": False,         # plot curv. linearization (original and solution based) (mincurv only)
             "raceline": True,                  # plot optimized path
             "imported_bounds": False,          # plot imported bounds (analyze difference to interpolated bounds)
             "raceline_curv": True,             # plot curvature profile of optimized path
             "racetraj_vel": True,              # plot velocity profile
             "racetraj_vel_3d": False,          # plot 3D velocity profile above raceline
             "racetraj_vel_3d_stepsize": 1.0,   # [m] vertical lines stepsize in 3D velocity profile plot
             "spline_normals": False,           # plot spline normals to check for crossings
             "mintime_plots": False}            # plot states, controls, friction coeffs etc. (mintime only)

# select track file (including centerline coordinates + track widths) --------------------------------------------------
# file_paths["track_name"] = "rounded_rectangle"                              # artificial track
# file_paths["track_name"] = "handling_track"                                 # artificial track
file_paths["track_name"] = "berlin_2018"                                    # Berlin Formula E 2018
# file_paths["track_name"] = "modena_2019"                                    # Modena 2019

# set import options ---------------------------------------------------------------------------------------------------
imp_opts = {"flip_imp_track": False,                # flip imported track to reverse direction
            "set_new_start": False,                 # set new starting point (changes order, not coordinates)
            "new_start": np.array([0.0, -47.0]),    # [x_m, y_m]
            "min_track_width": None,                # [m] minimum enforced track width (set None to deactivate)
            "num_laps": 1}                          # number of laps to be driven (significant with powertrain-option),
                                                    # only relevant in mintime-optimization

# set optimization type ------------------------------------------------------------------------------------------------
# 'shortest_path'       shortest path optimization
# 'mincurv'             minimum curvature optimization without iterative call
# 'mincurv_iqp'         minimum curvature optimization with iterative call
# 'mintime'             time-optimal trajectory optimization
opt_type = 'mintime'

# set mintime specific options (mintime only) --------------------------------------------------------------------------
# tpadata:                      set individual friction map data file if desired (e.g. for varmue maps), else set None,
#                               e.g. "berlin_2018_varmue08-12_tpadata.json"
# warm_start:                   [True/False] warm start IPOPT if previous result is available for current track
# var_friction:                 [-] None, "linear", "gauss" -> set if variable friction coefficients should be used
#                               either with linear regression or with gaussian basis functions (requires friction map)
# reopt_mintime_solution:       reoptimization of the mintime solution by min. curv. opt. for improved curv. smoothness
# recalc_vel_profile_by_tph:    override mintime velocity profile by ggv based calculation (see TPH package)

mintime_opts = {"tpadata": None,
                "warm_start": False,
                "var_friction": None,
                "reopt_mintime_solution": False,
                "recalc_vel_profile_by_tph": False}

# lap time calculation table -------------------------------------------------------------------------------------------
lap_time_mat_opts = {"use_lap_time_mat": False,             # calculate a lap time matrix (diff. top speeds and scales)
                     "gg_scale_range": [0.3, 1.0],          # range of gg scales to be covered
                     "gg_scale_stepsize": 0.05,             # step size to be applied
                     "top_speed_range": [100.0, 150.0],     # range of top speeds to be simulated [in km/h]
                     "top_speed_stepsize": 5.0,             # step size to be applied
                     "file": "lap_time_matrix.csv"}         # file name of the lap time matrix (stored in "outputs")

# ----------------------------------------------------------------------------------------------------------------------
# CHECK USER INPUT -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if opt_type not in ["shortest_path", "mincurv", "mincurv_iqp", "mintime"]:
    raise IOError("Unknown optimization type!")

if opt_type == "mintime" and not mintime_opts["recalc_vel_profile_by_tph"] and lap_time_mat_opts["use_lap_time_mat"]:
    raise IOError("Lap time calculation table should be created but velocity profile recalculation with TPH solver is"
                  " not allowed!")

# ----------------------------------------------------------------------------------------------------------------------
# CHECK PYTHON DEPENDENCIES --------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# get current path
file_paths["module"] = os.path.dirname(os.path.abspath(__file__))

# read dependencies from requirements.txt
requirements_path = os.path.join(file_paths["module"], 'requirements.txt')
dependencies = []

with open(requirements_path, 'r') as fh:
    line = fh.readline()

    while line:
        dependencies.append(line.rstrip())
        line = fh.readline()

# check dependencies
pkg_resources.require(dependencies)

# ----------------------------------------------------------------------------------------------------------------------
# INITIALIZATION OF PATHS ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# assemble track import path
file_paths["track_file"] = os.path.join(file_paths["module"], "inputs", "tracks", file_paths["track_name"] + ".csv")

# assemble friction map import paths
file_paths["tpamap"] = os.path.join(file_paths["module"], "inputs", "frictionmaps",
                                    file_paths["track_name"] + "_tpamap.csv")

if mintime_opts["tpadata"] is None:
    file_paths["tpadata"] = os.path.join(file_paths["module"], "inputs", "frictionmaps",
                                         file_paths["track_name"] + "_tpadata.json")
else:
    file_paths["tpadata"] = os.path.join(file_paths["module"], "inputs", "frictionmaps", mintime_opts["tpadata"])

# check if friction map files are existing if the var_friction option was set
if opt_type == 'mintime' \
        and mintime_opts["var_friction"] is not None \
        and not (os.path.exists(file_paths["tpadata"]) and os.path.exists(file_paths["tpamap"])):

    mintime_opts["var_friction"] = None
    print("WARNING: var_friction option is not None but friction map data is missing for current track -> Setting"
          " var_friction to None!")

# create outputs folder(s)
os.makedirs(file_paths["module"] + "/outputs", exist_ok=True)

if opt_type == 'mintime':
    os.makedirs(file_paths["module"] + "/outputs/mintime", exist_ok=True)

# assemble export paths
file_paths["mintime_export"] = os.path.join(file_paths["module"], "outputs", "mintime")
file_paths["traj_race_export"] = os.path.join(file_paths["module"], "outputs", "traj_race_cl.csv")
# file_paths["traj_ltpl_export"] = os.path.join(file_paths["module"], "outputs", "traj_ltpl_cl.csv")
file_paths["lap_time_mat_export"] = os.path.join(file_paths["module"], "outputs", lap_time_mat_opts["file"])

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT VEHICLE DEPENDENT PARAMETERS ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# load vehicle parameter file into a "pars" dict
parser = configparser.ConfigParser()
pars = {}

if not parser.read(os.path.join(file_paths["module"], "params", file_paths["veh_params_file"])):
    raise ValueError('Specified config file does not exist or is empty!')

pars["ggv_file"] = json.loads(parser.get('GENERAL_OPTIONS', 'ggv_file'))
pars["ax_max_machines_file"] = json.loads(parser.get('GENERAL_OPTIONS', 'ax_max_machines_file'))
pars["stepsize_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'stepsize_opts'))
pars["reg_smooth_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'reg_smooth_opts'))
pars["veh_params"] = json.loads(parser.get('GENERAL_OPTIONS', 'veh_params'))
pars["vel_calc_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'vel_calc_opts'))

if opt_type == 'shortest_path':
    pars["optim_opts"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_shortest_path'))

elif opt_type in ['mincurv', 'mincurv_iqp']:
    pars["optim_opts"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_mincurv'))

elif opt_type == 'mintime':
    pars["curv_calc_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'curv_calc_opts'))
    pars["optim_opts"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_mintime'))
    pars["vehicle_params_mintime"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'vehicle_params_mintime'))
    pars["tire_params_mintime"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'tire_params_mintime'))
    pars["pwr_params_mintime"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'pwr_params_mintime'))

    # modification of mintime options/parameters
    pars["optim_opts"]["var_friction"] = mintime_opts["var_friction"]
    pars["optim_opts"]["warm_start"] = mintime_opts["warm_start"]
    pars["vehicle_params_mintime"]["wheelbase"] = (pars["vehicle_params_mintime"]["wheelbase_front"]
                                                   + pars["vehicle_params_mintime"]["wheelbase_rear"])

# set import path for ggv diagram and ax_max_machines (if required)
if not (opt_type == 'mintime' and not mintime_opts["recalc_vel_profile_by_tph"]):
    file_paths["ggv_file"] = os.path.join(file_paths["module"], "inputs", "veh_dyn_info", pars["ggv_file"])
    file_paths["ax_max_machines_file"] = os.path.join(file_paths["module"], "inputs", "veh_dyn_info",
                                                      pars["ax_max_machines_file"])

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT TRACK AND VEHICLE DYNAMICS INFORMATION ------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# save start time
t_start = time.perf_counter()

# import track
reftrack_imp = helper_funcs_glob.src.import_track.import_track(imp_opts=imp_opts,
                                                               file_path=file_paths["track_file"],
                                                               width_veh=pars["veh_params"]["width"])

# import ggv and ax_max_machines (if required)
if not (opt_type == 'mintime' and not mintime_opts["recalc_vel_profile_by_tph"]):
    ggv, ax_max_machines = tph.import_veh_dyn_info.\
        import_veh_dyn_info(ggv_import_path=file_paths["ggv_file"],
                            ax_max_machines_import_path=file_paths["ax_max_machines_file"])
else:
    ggv = None
    ax_max_machines = None

# set ax_pos_safe / ax_neg_safe / ay_safe if required and not set in parameters file
if opt_type == 'mintime' and pars["optim_opts"]["safe_traj"] \
        and (pars["optim_opts"]["ax_pos_safe"] is None
             or pars["optim_opts"]["ax_neg_safe"] is None
             or pars["optim_opts"]["ay_safe"] is None):

    # get ggv if not available
    if ggv is None:
        ggv = tph.import_veh_dyn_info. \
            import_veh_dyn_info(ggv_import_path=file_paths["ggv_file"],
                                ax_max_machines_import_path=file_paths["ax_max_machines_file"])[0]

    # limit accelerations
    if pars["optim_opts"]["ax_pos_safe"] is None:
        pars["optim_opts"]["ax_pos_safe"] = np.amin(ggv[:, 1])
    if pars["optim_opts"]["ax_neg_safe"] is None:
        pars["optim_opts"]["ax_neg_safe"] = -np.amin(ggv[:, 1])
    if pars["optim_opts"]["ay_safe"] is None:
        pars["optim_opts"]["ay_safe"] = np.amin(ggv[:, 2])

# ----------------------------------------------------------------------------------------------------------------------
# PREPARE REFTRACK -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp = \
    helper_funcs_glob.src.prep_track.prep_track(reftrack_imp=reftrack_imp,
                                                reg_smooth_opts=pars["reg_smooth_opts"],
                                                stepsize_opts=pars["stepsize_opts"],
                                                debug=debug,
                                                min_width=imp_opts["min_track_width"])

bound_r = reftrack_interp[:, :2] + normvec_normalized_interp * np.expand_dims(reftrack_interp[:, 2], 1)
bound_l = reftrack_interp[:, :2] - normvec_normalized_interp * np.expand_dims(reftrack_interp[:, 3], 1)

plt.figure()
# plt.plot(refline[:, 0], refline[:, 1], "k--", linewidth=0.7)
# plt.plot(veh_bound1_virt[:, 0], veh_bound1_virt[:, 1], "b", linewidth=0.5)
# plt.plot(veh_bound2_virt[:, 0], veh_bound2_virt[:, 1], "b", linewidth=0.5)
# plt.plot(veh_bound1_real[:, 0], veh_bound1_real[:, 1], "c", linewidth=0.5)
# plt.plot(veh_bound2_real[:, 0], veh_bound2_real[:, 1], "c", linewidth=0.5)
plt.plot(bound_r[:, 0], bound_r[:, 1], "k-", linewidth=0.7)
plt.plot(bound_l[:, 0], bound_l[:, 1], "k-", linewidth=0.7)
# plt.plot(trajectory[:, 1], trajectory[:, 2], "r-", linewidth=0.7)

# if plot_opts["imported_bounds"] and bound1_imp is not None and bound2_imp is not None:
#     plt.plot(bound1_imp[:, 0], bound1_imp[:, 1], "y-", linewidth=0.7)
#     plt.plot(bound2_imp[:, 0], bound2_imp[:, 1], "y-", linewidth=0.7)

plt.grid()
ax = plt.gca()
# ax.arrow(point1_arrow[0], point1_arrow[1], vec_arrow[0], vec_arrow[1],
#             head_width=7.0, head_length=7.0, fc='g', ec='g')
ax.set_aspect("equal", "datalim")
plt.xlabel("east in m")
plt.ylabel("north in m")
plt.show()