import numpy as np
import matplotlib.pyplot as plt
import Bsm1Model_Scaled_No_p as Model
import mpctools as mpc
import random
import timeit
import itertools
import time
import pandas as pd
from tabulate import tabulate
from pandas import DataFrame
from RL_Toolbox import *
from Alarm_Pattern_Recognition import *
from Alarm_Prioritization import *

"""
Reward Function Area.
If set point is in the zone, return a fixed reward,
otherwise, reward is the difference between set point and actual
The fixed reward must be tuned to achieve optimal performance.
"""


def reward_calculator(eq, ntot, cod, snh_e, tss, bod, ae, pe, controller):
    rewards = 0
    "Reward for the KLa5 controller"
    if controller == 'kla5':
        if eq < 5130:
            rewards = (5130 - eq)
        else:
            rewards = -np.square(eq - 5130) - 50
        if ntot > 18:
            rewards = rewards - 50000
        if cod > 100:
            rewards = rewards - 50000
        if snh_e > 4:
            rewards = rewards - 50000
        if tss > 30:
            rewards = rewards - 50000
        if bod > 10:
            rewards = rewards - 50000

        if ae < 3480:
            rewards_ae = (3480 - ae) * 3.8
        else:
            rewards_ae = -np.square(ae - 3480) - 50

        rewards = rewards + rewards_ae
    # Reward for the Qa controller
    elif controller == 'qa':
        if eq < 5130:
            rewards = 5130 - eq
        else:
            rewards = -np.square(eq - 5130) - 50
        if ntot > 18:
            rewards = rewards - 50000
        if cod > 100:
            rewards = rewards - 50000
        if snh_e > 4:
            rewards = rewards - 50000
        if tss > 30:
            rewards = rewards - 50000
        if bod > 10:
            rewards = rewards - 50000

        if pe < 288:
            rewards_pe = (288 - pe) * 5
        else:
            rewards_pe = -np.square(pe - 288)

        rewards = rewards + rewards_pe

    return rewards


def pid(set_point, x_cur, x_1, gain, ti, td, u_1, error):

    ts = 0.001 * Time  # Sampling time
    ek = set_point - x_cur       # Current error
    ek_1 = set_point - x_1   # Error at time - 1
    error.append(ek)        # Add to the previous error

    ef = ek / (0.1*td + 1)

    "Discrete Time PID, derivative part is not accurate.  Ef should be at (k - 1) and (k - 2),"
    "but we don't use the derivative part"
    du = gain * ((ek - ek_1) + ts * ek / ti) + (gain * td) / ts * (ef - 2*ef + ef)
    u_cur = u_1 + du

    return u_cur


"""
Alarm Section.  If any of the following is at 90% of the threshold, HH alarm triggers.
At 75% of capacity, H alarm triggers.
Everything is divided by 2 because the args input is an Alarm and its corresponding set point.
"""


def alarm(plant_alarms, *args):
    if plant_alarms.shape[0] < len(args) / 2:
        plant_alarms = np.r_[plant_alarms, np.zeros((int(len(args) / 2) - plant_alarms.shape[0],
                                                     plant_alarms.shape[1]))]

    for l in range(0, len(args), 2):
        if args[l + 1] * 0.75 < args[l] <= args[l + 1] * 0.9:
            plant_alarms[int(l / 2), j] = 1
        elif args[l] > args[l + 1] * 0.9:
            plant_alarms[int(l / 2), j] = 2

    return plant_alarms


"""
This section is used for alarm sequence generation or appending to existing alarm sequences.

alarms: The alarm log from the plant, in numbers
number_of_alarms: The number of alarms in the plant
old_seq_dict: The old sequence dictionary
old_seq_num: The old sequence mapping from sequences to alarm numbers
old_rev_dict: The old sequence number mapping from seq number to alarm sequence
gen_new: True = Generate brand new alarm sequence, False = append current alarms to existing sequence

output_seq_dict: The new sequence dictionary after generating new / appending alarms
output_seq_numbers: The new sequence numbers after generating new / appending alarms
output_rev_dict: The new reverse dictionary after generating new / appending alarms
"""


def alarm_sequence_generation(alarms, number_of_alarms, old_seq_dict, old_seq_num,
                              old_rev_dict, gen_new):

    output_seq_dict = "Empty"
    output_seq_numbers = "Empty"
    output_rev_dict = "Empty"

    if gen_new is False:
        new_seq_dict, new_seq_num, new_rev_dict = sequence_dict_editor(alarms, number_of_alarms, old_seq_dict, "N/A",
                                                                       old_seq_num, "N/A", old_rev_dict, "N/A",
                                                                       gen_new=True, save=True)

        output_seq_dict = alarm_appender(old_seq_dict, new_seq_dict)
        output_seq_numbers = alarm_appender(old_seq_num, new_seq_num)
        output_rev_dict = alarm_appender(old_rev_dict, new_rev_dict)

    elif gen_new is True:
        output_seq_dict, output_seq_numbers, output_rev_dict = sequence_dict_editor(alarms, number_of_alarms, _, _, _,
                                                                                    _, _, _, gen_new=True, save=True)

    else:
        print("Error in specifying")

    return output_seq_dict, output_seq_numbers, output_rev_dict


"""
Epsilon calculation for ε-greedy policy.  Random action at eps = (1 - ε)%
"""


def rl_action(state, action, action_list, control_list, q_list, nt, egreedy):

    if egreedy is True:
        if nt[state, action] < 10:
            eps = 0.5
        else:
            eps = 1 - (0.5 * 1 / (1 + np.sqrt(nt[state, action])))

        eps = min(eps, 0.7)
    else:
        eps = 1

    "Epsilon Greedy Action"

    number = np.random.rand()
    if number < eps:
        action = rargmax(q_list)
    else:
        action = random.randint(0, len(action_list) - 1)

    "Take the action"
    if j == 0:
        "If this is the first time step"
        control = control_list[j] + action_list[action]
        control = max(control, 0)
    else:
        control = control_list[j - 1] + action_list[action]
        control = max(control, 0)

    return control, action


"""
Initialize the action and state spaces.
Then initialize the Q matrix.
"""

actions = list(np.zeros(16))
actions[0:15] = np.arange(-0.5, 0.5, 1 / (len(actions))*0.995)

states_kla5 = list(np.zeros(65))
states_kla5[0:65] = np.arange(0.35, 2.35, 2 / len(states_kla5)*0.997)

states_qa = list(np.zeros(65))
states_qa[0:65] = np.arange(1, 3, 2 / len(states_qa)*0.997)

Q_kla5 = np.loadtxt("Qmatrix_Autosave_KLa5.txt")
Q_qa = np.loadtxt("Qmatrix_Autosave_Qa.txt")

"""
Upper Confidence Bound initialization for matrices and degrees of exploration.
"""

nt_kla5 = np.loadtxt("NTmatrix_Autosave_KLa5.txt")
t_kla5 = np.loadtxt("tmatrix_Autosave_KLa5.txt")

nt_qa = np.loadtxt("NTmatrix_Autosave_Qa.txt")
t_qa = np.loadtxt("tmatrix_Autosave_Qa.txt")

c = 0

"""
The discount factor, number of iterations, and how often RL evaluates.
"""

discount_factor = 0.97
NumEpisodes = 5
RL_Evaluate = 100    # In minute scale

"""
Model Disturbance and parameter loading.
Initialize model matrices.
"""

rList = []
data = np.loadtxt('Inf_dry_2006_split_60s.txt')
Time = 1/3        # 1 is equal to 15 minutes
open_ss_bsm1 = np.loadtxt('ss_op.txt').T

Z0_14 = data[1:14, :]  # Disturbances
Q0_14 = data[14, :]  # Disturbances

Delta = 1/15  # To break the simulation in discrete simulator
Nsim = data.shape[1]  # 1345 Time steps
Nx = open_ss_bsm1.shape[0]  # Number of states
# Nu = data.shape[0] + 1  # To test, Comment,  Number of inputs
Nu = 2                             # To test, Comment

"""
Run the model for the NumEpisodes iterations.
Hope that the Q-Matrix will learn properly and converge

***The current model runs at steady state and with no real weather data.  To run the simulation using real water data:
1. Uncomment line 246.
2. Comment line 247.
3. Delete the "_constant_distur" portion in line 288.
4. Uncomment lines 290 and 291.
"""

for i in range(1, NumEpisodes):

    """
    Load the dictionaries of the alarms and initialization of some variables used for pattern recognition.
    """

    Seq_dictionary = load_obj("Seq_Dictionary")
    Seq_dictionary_numbers = load_obj("Seq_Dictionary_Numbers")
    Rev_dictionary = load_obj("Reverse_Dictionary")

    key = "none"
    placeholder = []
    sequence_length = 0
    alarms_in_plant = []
    masked_alarm_log = []

    Alarms = np.zeros((1, Nsim + 1))    # Alarm Matrix
    alarm_pri_matrix = np.array([["Alarms"], ["Optimal RL Values"]])
    value_sequence_dict = {}
    length_keymaker = 0

    """
    Simulation Characteristics
    """

    x = np.zeros((Nx, Nsim + 1))        # States, Simulation Time
    u = np.zeros((Nu, Nsim))            # Inputs, Simulation Time

    wwtp_sim = mpc.DiscreteSimulator(Model.ode_bsm1model_constant_distur, Delta, [Nx, Nu], ["x", "u"])

    # u[3:16, :] = Z0_14                  # To test, Comment
    # u[2, :] = Q0_14                     # To test, Comment

    x0 = open_ss_bsm1                  # Load initial states

    """
    Initiate parameters as zero
    """

    Q1_14 = np.zeros(Nsim)
    Qe_14 = np.zeros(Nsim)
    Qf_14 = np.zeros(Nsim)
    Qw_14 = np.zeros(Nsim)
    Qr_14 = np.zeros(Nsim)
    Qa_14 = np.zeros(Nsim)
    KLa5_14 = np.zeros(Nsim)
    r_so_14 = np.zeros(Nsim)
    r_sno_14 = np.zeros(Nsim)

    r_so = 0.9
    r_sno = 1.9
    Qr = 18446
    Qw = 385
    Q0_stab = 18446
    KLa5 = 131.65

    r_so_14[0] = r_so
    r_sno_14[0] = r_sno
    Qa_14[0] = 16485.6074
    KLa5_14[0] = 131.65
    x[:, 0] = x0
    x[59, 0] = r_so
    x[21, 0] = r_sno

    """""
    Initiate Lists
    """""

    error1 = []
    error2 = []

    Ntot_list = []
    COD_list = []
    Snh_list = []
    TSS_list = []
    BOD_list = []

    TSSa_list = []
    Xw_list = []

    EQ_list = []
    PE_list = []
    AE_list = []
    ME_list = []
    OCI_list = []

    IAE_sno1 = []
    IAE_so1 = []

    ISE_sno1 = []
    ISE_so1 = []

    control_action_Qa = []
    control_action_KLa5 = []

    reward_list = []

    """
    Initiate rewards, states, actions, and EQ and OCI
    """

    r = 0
    s = 0
    a = 0
    PE = 0
    AE = 0
    EQ = 0
    OCI = 0
    s_kla5 = 0
    a_kla5 = 0
    s_qa = 0
    a_qa = 0

    # Random dummy value to bypass NameError for undefined value.
    feedback_evaluate = 995

    """
    Visualization Tools for Code. 
    
    Comment out lines 379 - 403 and 863 - 885 to remove the live visualization.  The visualization may appear laggy.
    This is because the code is not complied before execution (i.e., this is a script).
    """

    # # R_so visualizations
    # x_plant = np.array([[0], [0]])
    # x_rl_rso = np.array([[0], [0]])
    #
    # plt.ion()
    # actual = plt.plot(x_plant[0, :], x_plant[1, :])[0]
    # rl_setpoint = plt.plot(x_rl_rso[0, :], x_rl_rso[1, :])[0]
    #
    # # R_sno visualizations
    # x_plant_rsno = np.array([[0], [0]])
    # x_rl_rsno = np.array([[0], [0]])
    #
    # actual_rsno = plt.plot(x_plant_rsno[0, :], x_plant_rsno[1, :])[0]
    # rl_setpoint_rsno = plt.plot(x_rl_rsno[0, :], x_rl_rsno[1, :])[0]
    #
    # plt.ylim(0, 2.5)
    # plt.xlim(0, 14)
    # plt.xlabel("Time, (Days)")
    # plt.ylabel("Set point")
    # plt.legend([actual, rl_setpoint, actual_rsno, rl_setpoint_rsno],
    #            ['Plant r_so', 'RL r_so Recommendation', 'Plant r_sno', 'RL r_sno Recommendation'])
    #
    # # Alarm Table
    # headers = ["Chronological", "Alarm Sequence", "VPC Score"]
    # last_alarm_log = []

    """
    Simulation Initiation.  14 days.
    """

    for j in range(Nsim):

        """
        Weather and Flow Rate data
        """

        Z0 = Z0_14[:, j]                # Get weather data at each time instant
        Q0 = Q0_14[j]                   # Get flow rate at each time instance
        Qa = 16485.6074                 # When Qa is commented out to only deal with KLa5

        """
        The implementation of reinforcement learning.
        Slices the Q matrix into a list, then finds the max in the list
        """

        if j % RL_Evaluate == 0 and j != 0:

            "Time step to evaluate feedback for RL."
            feedback_evaluate = (j - 1) + RL_Evaluate

            """
            Reinforcement Learning: State Detection
            """

            # KLa5 Controls
            x_curr_kla5 = x[59, j]
            s_kla5 = min(states_kla5, key=lambda x_current: abs(x_current - x_curr_kla5))
            "Return index of the current state"
            s_kla5 = states_kla5.index(s_kla5)

            Q_list_kla5 = Q_kla5[s_kla5, :].tolist()

            for Action in range(len(Q_list_kla5)):
                Q_list_kla5[Action] = Q_list_kla5[Action] + c * np.sqrt(np.log(t_kla5[s_kla5, Action]) /
                                                                        (nt_kla5[s_kla5, Action] + 0.01))

            a_kla5 = rargmax(Q_list_kla5)

            r_so, a_kla5 = rl_action(s_kla5, a_kla5, actions, r_so_14, Q_list_kla5, nt_kla5, False)

            # Qa Controls
            x_curr_qa = x[21, j]
            s_qa = min(states_qa, key=lambda x_current: abs(x_current - x_curr_qa))
            "Return index of the current state"
            s_qa = states_qa.index(s_qa)

            Q_list_qa = Q_qa[s_qa, :].tolist()

            for Action in range(len(Q_list_qa)):
                Q_list_qa[Action] = Q_list_qa[Action] + c * np.sqrt(np.log(t_qa[s_qa, Action]) / (nt_qa[s_qa, Action]
                                                                                                  + 0.01))

            a_qa = rargmax(Q_list_qa)

            r_sno, a_qa = rl_action(s_qa, a_qa, actions, r_sno_14, Q_list_qa, nt_qa, False)

        """
        Proportional, Integral, Derivative controllers for Qa and KLa5
        """

        if j == 0:
            Qa = pid(r_sno, x[21, j], r_sno, 10000, 0.00167, 0.0, Qa_14[0], error1)
            KLa5 = pid(r_so, x[59, j], r_so, 25, 0.002, 0.0, KLa5_14[0], error2)
        else:
            Qa = pid(r_sno, x[21, j], x[21, j - 1], 10000, 0.00167, 0.0, Qa_14[j - 1], error1)
            KLa5 = pid(r_so, x[59, j], x[59, j - 1], 25, 0.002, 0.0, KLa5_14[j - 1], error2)

        control_action_Qa.append(Qa)
        control_action_KLa5.append(KLa5)

        """
        Introduce Disturbances to generate alarms
        1. High Soluble Inert Organic Matter in Inlet.
        2. Tanks 1 - 5 did not remove sufficient Nitrogen.
        """

        if 200 < j < 205:
            x[0, j] = 60

        if 500 < j < 505:
            x[0, j] = 60

        if 900 < j < 905:
            x[0, j] = 60

        if 2900 < j < 2905:
            x[0, j] = 60

        if 5900 < j < 5905:
            x[0, j] = 60

        if 1000 < j < 1050:
            x[61, j] = 10

        if 2000 < j < 2050:
            x[61, j] = 10

        if 3000 < j < 3050:
            x[61, j] = 10

        if 4000 < j < 4050:
            x[61, j] = 10

        """
        Housekeeping to ensure Qa and KLa5 are within physically possible levels.
        """

        if Qa < 0:
            Qa = 0
        elif Qa > 5*Q0_stab:
            Qa = 5*Q0_stab

        if KLa5 < 0:
            KLa5 = 0
        elif KLa5 > 240:
            KLa5 = 240

        Q1 = Q0 + Qa + Qr               # Inlet into plant
        Qe = Q0 - Qw                    # Flow rate to river

        Qf = Q1 - Qa                    # Flow to Settler

        Q1_14[j] = Q1                   # Flow into system
        Qe_14[j] = Qe                   # Flow to river
        Qf_14[j] = Qf                   # Flow to settler
        Qw_14[j] = Qw                   # Waste water
        Qr_14[j] = Qr                   # External Recycle
        Qa_14[j] = Qa                   # Internal recycle
        KLa5_14[j] = KLa5               # Mass Transfer Coefficient
        r_so_14[j] = r_so               # Oxygen in tank 5
        r_sno_14[j] = r_sno             # Nitrogen in tank 2

        """
        Performance Assessment Section.
        All performance assessment regarding the waste water treatment plant
        are calculated here.  The subscript e represents effluent.
        """

        Xe = x[74, j]
        Xf = 0.75 * np.sum(x[54:59, j])
        Multiplier = Xe / Xf

        "In-Soluble Effluent Qualities"

        Xi_e = Multiplier * x[54, j]
        Xs_e = Multiplier * x[55, j]
        Xbh_e = Multiplier * x[56, j]
        Xba_e = Multiplier * x[57, j]
        Xp_e = Multiplier * x[58, j]
        Xnd_e = Multiplier * x[63, j]

        "Soluble Effluent Qualities"

        Si_e = x[84, j]
        Ss_e = x[94, j]
        So_e = x[104, j]
        Sno_e = x[114, j]
        Snh_e = x[124, j]
        Snd_e = x[134, j]
        Salk_e = x[144, j]

        Snk_e = Snh_e + Snd_e + Xnd_e + 0.08 * (Xbh_e + Xba_e) + 0.06 * (Xp_e + Xi_e)
        Ntot = Snk_e + Sno_e
        COD = Ss_e + Si_e + Xs_e + Xi_e + Xbh_e + Xba_e + Xp_e
        BOD = 0.25 * (Ss_e + Xs_e + (1 - 0.08) * (Xbh_e + Xba_e))

        SSE = 0.75 * (Xi_e + Xs_e + Xbh_e + Xba_e + Xp_e)
        EQc = (2 * Xe + COD + 30*Snk_e + 10*Sno_e + 2*BOD)*Qe / 1000

        EQ_list.append(EQc)

        # List of all the terrible stuff
        Ntot_list.append(Ntot)
        COD_list.append(COD)
        Snh_list.append(Snh_e)
        TSS_list.append(Xe)
        BOD_list.append(BOD)

        "Waste water Characteristics"
        X_w = (x[65, j] * Qw) / 1000
        Xw_list.append(X_w)

        """
        The amount of sludge production to be disposed.
        """

        if j == int(round(Nsim*0.5)) or j == (Nsim - 1):
            TSSa = 0.75 * np.sum(np.sum(x[2:7, j])*1000       # Total TSS
                                 + np.sum(x[15:20, j])*1000
                                 + np.sum(x[28:33, j])*1333
                                 + np.sum(x[41:46, j])*1333
                                 + np.sum(x[54:59, j])*1333)

            TSSs = 0
            for iTSS in range(65, 75):
                TSSs += 0.75 * x[iTSS, j] * 1500 * 0.4
            else:
                TSSs = TSSs

            TSS = (TSSa + TSSs) / 1000
            TSSa_list.append(TSS)

        """
        Pumping energy required for internal and external flow recycle pumps.
        """
        PEc = 0.004 * Qa + 0.008 * Qr + 0.05 * Qw
        PE_list.append(PEc)

        """
        Aeration energy required for the plant.  Assumed So = 8, Kla3, 4 = 240
        """

        AEc = (8 / (1.8 * 1000)) * np.sum(1333*240*2 + 1333*KLa5)
        AE_list.append(AEc)

        """
        Mixing Energy Consumption
        """

        ME = 24*0.005*1000*2
        ME_list.append(ME)

        "Dynamic OCI, including only pumping energy and aeration energy"

        OCI_list.append(PEc + AEc)

        _ = control_error("IAE", r_sno, x[21, j], IAE_sno1)
        _ = control_error("IAE", r_so, x[59, j], IAE_so1)

        _ = control_error("ISE", r_sno, x[21, j], ISE_sno1)
        _ = control_error("ISE", r_so, x[59, j], ISE_so1)

        """
        Calculate the next time step's states using the discrete simulator.  
        """

        u[0, :] = KLa5_14
        u[1, :] = Qa_14

        x[:, j + 1] = wwtp_sim.sim(x[:, j], u[:, j])

        """
        Total Alarms: 20
        Alarm triggers for Ntot, COD, Snh_e, Xe and BOD, 
        Si in tanks 1 - 5
        Snh in separator layers 1 - 10
        KLa5 set point and actual.  Perhaps KLa is broken.
        """

        Alarms = alarm(Alarms, Ntot, 22, COD, 100, Snh_e, 4, Xe, 30, BOD, 10,
                       x[0, j], 45, x[13, j], 45, x[26, j], 45, x[39, j], 45, x[52, j], 45,
                       x[120, j], 3, x[119, j], 3, x[121, j], 3, x[118, j], 3, x[122, j], 3,
                       x[117, j], 3, x[123, j], 3, x[116, j], 3, x[124, j], 3, x[115, j], 3)

        """
        Generate the original plant alarm log and the masked alarm log.  Masked alarm log will convey equal information
        as the original plant alarm log, however, it will e tremendously smaller.
        """

        alarms_in_plant, masked_alarm_log, placeholder, sequence_length, key = live_alarm_log(alarms_in_plant,
                                                                                              masked_alarm_log,
                                                                                              Alarms,
                                                                                              Seq_dictionary,
                                                                                              Seq_dictionary_numbers,
                                                                                              21,
                                                                                              j,
                                                                                              placeholder,
                                                                                              sequence_length,
                                                                                              key,
                                                                                              Rev_dictionary)

        """
        Generate the alarm priority matrices.  Events that results in lower Q-values will be placed on top as higher
        priority alarms.
        """

        if j >= 1:
            x_temp_kla5 = x[59, j + 1]
            s_temp_kla5 = min(states_kla5, key=lambda x_current: abs(x_current - x_temp_kla5))
            s_temp_kla5 = states_kla5.index(s_temp_kla5)

            x_temp_qa = x[21, j + 1]
            s_temp_qa = min(states_qa, key=lambda x_current: abs(x_current - x_temp_qa))
            s_temp_qa = states_qa.index(s_temp_qa)

            "Did not add Qa values"
            alarm_pri_matrix = alarm_prioritization(alarm_pri_matrix, Alarms, max(Q_kla5[s_temp_kla5, :]),
                                                    -(Alarms.shape[1] - j))

            "Assigns a RL number to each alarm sequence"
            value_sequence_dict, length_keymaker = key_maker(alarm_pri_matrix, Seq_dictionary,
                                                             value_sequence_dict, length_keymaker, 0.5, "exp")

            """
            Visualization of alarm table
            """

            # if the alarm log was identical to last time step, skip evaluation.
            # if last_alarm_log == alarms_in_plant or len(alarm_pri_matrix[0]) == 1:
            #     pass
            #
            # else:
            #     "Initialize the list as the length of the masked alarm log"
            #     alarm_matrix = np.array([masked_alarm_log, np.zeros(len(masked_alarm_log))])
            #
            #     "Find all alarms and warnings that are not sequences"
            #     for k in range(len(masked_alarm_log)):
            #         if alarm_matrix[0, k][0:8] != "Sequence":
            #
            #             temp_key = alarm_matrix[0, k].split()
            #             if temp_key[0] == "Alarm":
            #                 tempo_key = "A" + str(temp_key[1])
            #             else:
            #                 tempo_key = "W" + str(temp_key[1])
            #
            #             "Find the Q-value for the individual alarm"
            #             for q in range(alarm_pri_matrix.shape[1]):
            #                 if tempo_key == alarm_pri_matrix[0, -q]:
            #                     "The divided by 1250 is a scaling factor to get values smaller.  It is arbitrary"
            #                     alarm_matrix[1, k] = float(alarm_pri_matrix[1, q]) / 1250
            #                 else:
            #                     pass
            #
            #             "If the alarm is a sequence"
            #         elif alarm_matrix[0, k][0:8] == "Sequence":
            #             # Get the sequence number
            #             rev_key = int(alarm_matrix[0, k].split()[-1])
            #             # Get the sequence of alarms generating that sequence
            #             Sequence_key = Rev_dictionary[rev_key]
            #             # Take that sequence key and use it in value_sequence_dictionary
            #             Q_score = value_sequence_dict[Sequence_key]
            #             "The divided by 2500 is a scaling factor to get values smaller.  It is arbitrary"
            #             alarm_matrix[1, k] = Q_score / 2500
            #
            #         else:
            #             print("Error in Alarm Table")
            #
            #     alarm_matrix = alarm_matrix.T
            #
            #     "Using Pandas library to sort the table"
            #     df = pd.DataFrame({
            #         headers[1]: alarm_matrix[:, 0],
            #         headers[2]: alarm_matrix[:, 1],
            #     })
            #
            #     df = df.sort_values(by='VPC Score')
            #     print(tabulate(df, headers=headers[0:3]))
            #
            #     time.sleep(1)
            #
            #     last_alarm_log = alarms_in_plant.copy()

        """
        Calculates the TSS (Total Suspended Solids) over the last 7 days.
        """

        if j == (Nsim - 1):
            eval_period = 14
            EQ = (1 / eval_period) * trapezoid(0, eval_period, len(EQ_list) - 1, EQ_list)
            PE = (1 / eval_period) * trapezoid(0, eval_period, len(PE_list) - 1, PE_list)
            AE = (1 / eval_period) * trapezoid(0, eval_period, len(AE_list) - 1, AE_list)
            SP = (1 / eval_period) * ((TSSa_list[1] - TSSa_list[0]) + trapezoid(0, eval_period, len(Xw_list) - 1,
                                                                                Xw_list))
            ME = (1 / eval_period) * trapezoid(0, eval_period, len(ME_list) - 1, ME_list)
            IAE_sno = trapezoid(0, 14, len(IAE_sno1) - 1, IAE_sno1)
            IAE_so = trapezoid(0, eval_period, len(IAE_so1) - 1, IAE_so1)

            ISE_sno = trapezoid(0, 14, len(ISE_sno1) - 1, ISE_sno1)
            ISE_so = trapezoid(0, eval_period, len(ISE_so1) - 1, ISE_so1)

            OCI = (PE + AE + 5*SP + ME)

        """
        Returns reward, new state, and the action taken.
        EQ is normalized by Qe because Qe cannot be controlled.
        This portion of the code must always be 1 step behind the above step.
        
        The above RL observes the current state and performs an action.  This portion assumes the previous action
        took the plant to a new state, and observes the reward and the new state and judges how good the previous 
        state-action pair was.
        """

        if j == feedback_evaluate:

            scale = 18061 / Qe_14[j]

            reward_kla5 = reward_calculator(EQ_list[j - 1] * scale, Ntot_list[j - 1], COD_list[j - 1], Snh_list[j - 1],
                                            TSS_list[j - 1], BOD_list[j - 1], AE_list[j - 1], PE_list[j - 1],
                                            controller='kla5')

            reward_qa = reward_calculator(EQ_list[j - 1] * scale, Ntot_list[j - 1], COD_list[j - 1], Snh_list[j - 1],
                                          TSS_list[j - 1], BOD_list[j - 1], AE_list[j - 1], PE_list[j - 1],
                                          controller='qa')

            x_next_kla5 = r_so_14[j - 1]
            s1_kla5 = min(states_kla5, key=lambda x_current: abs(x_current - x_next_kla5))
            s1_kla5 = states_kla5.index(s1_kla5)

            x_next_qa = r_sno_14[j - 1]
            s1_qa = min(states_qa, key=lambda x_current: abs(x_current - x_next_qa))
            s1_qa = states_qa.index(s1_qa)

            """
            Reinforcement Learning: Learning Rate
            """

            if nt_kla5[s_kla5, a_kla5] <= 15:
                learning_rate_kla5 = 0.5
            else:
                learning_rate_kla5 = 0.5 / (nt_kla5[s_kla5, a_kla5] - 14)

            learning_rate_kla5 = max(learning_rate_kla5, 0.002)

            # learning_rate_kla5 = 0

            if nt_qa[s_qa, a_qa] <= 15:
                learning_rate_qa = 0.5
            else:
                learning_rate_qa = 0.5 / (nt_qa[s_qa, a_qa] - 14)

            learning_rate_qa = max(learning_rate_qa, 0.002)

            # learning_rate_qa = 0

            "Update the Q-Table with new values"

            Q_kla5[s_kla5, a_kla5] = Q_kla5[s_kla5, a_kla5] + learning_rate_qa*(reward_kla5 + discount_factor *
                                                                                np.max(Q_kla5[s1_kla5, :]) -
                                                                                Q_kla5[s_kla5, a_kla5])

            Q_qa[s_qa, a_qa] = Q_qa[s_qa, a_qa] + learning_rate_qa * (reward_qa + discount_factor *
                                                                      np.max(Q_qa[s1_qa, :]) - Q_qa[s_qa, a_qa])
            r += reward_kla5

            """
            UCB matrices update
            """

            nt_kla5[s_kla5, a_kla5] = nt_kla5[s_kla5, a_kla5] + 1

            for k in range(t_kla5.shape[1]):
                if k != a_kla5:
                    t_kla5[s_kla5, k] = t_kla5[s_kla5, k] + 1
                else:
                    pass

            nt_qa[s_qa, a_qa] = nt_qa[s_qa, a_qa] + 1

            for k in range(t_qa.shape[1]):
                if k != a_qa:
                    t_qa[s_qa, k] = t_qa[s_qa, k] + 1
                else:
                    pass

            """
            Live Updating.  Comment this part out with the 
            """

            # # R_so visualizations
            # x_plant = np.append(x_plant, [[j / 1440], [x[59, j]]], axis=1)
            # actual.set_xdata(x_plant[0, :])
            # actual.set_ydata(x_plant[1, :])
            #
            # x_rl_rso = np.append(x_rl_rso, [[j / 1440], [r_so_14[j]]], axis=1)
            # rl_setpoint.set_xdata(x_rl_rso[0, :])
            # rl_setpoint.set_ydata(x_rl_rso[1, :])
            #
            # plt.draw()
            # plt.pause(0.01)
            #
            # # R_sno visualizations
            # x_plant_rsno = np.append(x_plant_rsno, [[j / 1440], [x[21, j]]], axis=1)
            # actual_rsno.set_xdata(x_plant_rsno[0, :])
            # actual_rsno.set_ydata(x_plant_rsno[1, :])
            #
            # x_rl_rsno = np.append(x_rl_rsno, [[j / 1440], [r_sno_14[j]]], axis=1)
            # rl_setpoint_rsno.set_xdata(x_rl_rsno[0, :])
            # rl_setpoint_rsno.set_ydata(x_rl_rsno[1, :])
            #
            # plt.draw()
            # plt.pause(0.01)

    rList.append(r)

    print("The Effluent Quality, Aeration Energy, Pumping Energy and Overall Cost Index are: %s, %s, %s and %s." %
          (EQ, AE, PE, OCI))

    if i % 100 == 0 and i != 0:
        print("Saving...")
        np.savetxt("Qmatrix_Autosave_KLa5.txt", Q)
        np.savetxt("NTmatrix_Autosave_KLa5.txt", nt)
        np.savetxt("tmatrix_Autosave_KLa5.tsequence_dxt", t)
