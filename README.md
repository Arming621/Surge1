Python Code

from __future__ import print_function
import pandas
from scipy.optimize import NonlinearConstraint, LinearConstraint
from scipy.optimize import BFGS, minimize, Bounds, SR1
from scipy.optimize import minimize_scalar
import numpy
import numpy as np
from pathlib import Path
import os.path
import time
import os
import math
import pyipopt
from numpy import *
P = np.array([[1060, 22], [22, 0.52]])

########################################### steady-state ###################################
CAs= 1.9537
Ts=  401.8727
def MPC_SIMULATION(CAi,Ti):
    
    ################################### Simulation time step ##################################
    delta = 0.01   # sampling time
    hc = 1e-4     # integration time step
    oper_time = 0.01  

    ################################## Initial states #########################################

    x1_record=[CAi]
    x2_record=[Ti]
    x2_sensor_record=[Ti]
    u1_record=[]
    u2_record=[]

    ##################################### Constants ###########################################
    a=1060
    b=22
    d=0.52  # Lyapunov function V=x^T*P*x


    # CSTR PARAMETERS
    F=5
    V=1
    k0=8460000
    E=50000     # parametric drift #####E has the most effect on process dynamics######
    R=8.314
    T0=300
    Dh=-11500
    rho=1000
    sigma=1000
    cp=0.231
    Qs=0
    CA0s=4

    ########################################### steady-state ###################################

    # state_ss=numpy.array([Ts, CAs])
    # input_ss=numpy.array([Qs, CA0s])
    state_ss=numpy.array([CAs,Ts])
    input_ss=numpy.array([CA0s,Qs])

    ROOT_FOLDER=os.getcwd()

    ##################################### MPC Parameters ######################################

    NUM_MPC_ITERATION=10   # MPC TOTAL ITERATION


    NUM_OUTPUTS = 2 # Number of state variables (RNN output)  
    NUM_INPUTS = 2  # Number of manipulated input
    HORIZON = 2   ## MPC PREDICTION HORIZON (Depends on how many delta you want to predict with the RNN)

    
    NUM_IN_SEQUENCE = 10  # Number of integration time steps actually used in MPC (equal to timestep of ML model)
    PREDICTION_STORE = 0   ## 

    NUM_MPC_INPUTS = NUM_INPUTS*HORIZON  # Number of MPC inputs to be optimized 
    NUM_MPC_CONSTRAINTS = HORIZON  # For each sampling time within prediction horizon, we have 1 constraint

    realtime_data = None

    setpoint=[0, 0]

    def my_ens_prediction(num_horizon,my_rawdata,my_inputs): 
        '''
        my_rawdata: current state variables CA, T; 
        my_rawdata = realtime_data;
        my_inputs: Manipulated variables u1, u2 to be optimzed; u1 = CA0 , u2 = Q;
        my_inputs[0]: u1(t=0), my_inputs[1]: u2(t=0) ; my_inputs[2]=u1(t=1), my_inputs[3]=u1(t=1)
        '''

        ensemble_output = numpy.zeros((num_horizon,NUM_IN_SEQUENCE,NUM_OUTPUTS)) # Initialize the prediction output matrix
        
        predict_output = []
        x_test2 = my_rawdata[0:NUM_OUTPUTS].astype(float)  ## CA, T
        #x_test2= (x_test2+state_ss-state_mean)/state_std

        # predict_output_normal=[[0 for i in range(NUM_OUTPUTS)] for j in range(NUM_IN_SEQUENCE)] ## INITALIZATION
        '''
        Generally, we let CA be first state [0], and T be the second state [1]. 
        '''
        x1=x_test2[0]  ## CA
        x2=x_test2[1]  ## T

        for i_model in range(num_horizon):  ## Explicit euler 

            for kk in range (int(delta/hc)):
        
                x1_new = x1 + hc * ((F / V) * (my_inputs[2*i_model] - x1) -
                                k0 * ((numpy.exp(-E / (R * (x2 + Ts)))*(x1 + CAs) * (x1 + CAs))
                                    - numpy.exp(-E / (R * Ts)) * CAs * CAs))

                x2_new = x2 + hc * (((F / V) * (-x2) + (-Dh / (sigma * cp)) *
                                (k0 * ((numpy.exp(-E / (R * (x2 + Ts))) * (x1 + CAs) * (x1 + CAs)) -
                                        numpy.exp(-E / (R * Ts)) * CAs * CAs)) + (my_inputs[2*i_model+1] / (sigma * cp * V))))


                x1 = x1_new
                x2 = x2_new
                if (kk % 10 == 0):  #   delta/hc points, saved every 10 points
                    predict_output = [x1, x2]
                    ensemble_output[i_model, int(kk / 10), 0: 2] = predict_output


        return ensemble_output    

    #################################################
    ################## MPC PROGRAM ##################
    #################################################
    ### DEFINE THE UPPER BOUND AND LOWER BOUND OF THE MANIPULATED INPUTS ###

    def eval_f(x):    
        '''
        Define objective function
        '''
        assert len(x) == int(NUM_MPC_INPUTS)
        offset=0
        global PREDICTION_STORE
        ############################################# CALCULATE OUTLET CONC #########################################################
        df_ensemble_output = my_ens_prediction(num_horizon = HORIZON,my_rawdata=realtime_data,my_inputs=x)
    
        # factor is used to balance the contribution of x and u in MPC objective func.; we can also pick constants
        factor = realtime_data[1] ** 2 * 500 / (realtime_data[0] ** 2 * 50)
        factor2 = realtime_data[1] ** 2 * 500 / (3.5 ** 2 * 10000)

        ############################################### Account for all intermediate steps ###########################################
        for j in range (HORIZON):
            est_outlet_product = df_ensemble_output[j, :, 0:2]
            for i in range (NUM_IN_SEQUENCE):
                #offset = (setpoint[0]-(est_outlet_product[i,0])) ** 2.0 *2 +(setpoint[1]-(est_outlet_product[i,1])) ** 2.0 *500 +\
                #    x[2*j] **2 *5e-6 + 0.6* x[2*j+1] **2
                offset = offset + (setpoint[0]-(est_outlet_product[i,1])) ** 2.0 *0.5 + (setpoint[1]-(est_outlet_product[i,0])) ** 2.0 *500 +\
                    0#x[2*j] **2 *1e-6 + 0.6* x[2*j+1] **2
            #offset = offset + x[2 * j] ** 2 * factor2 * 3e-10 + factor2 * x[2 * j + 1] ** 2



        return offset/100

    def eval_grad_f(x):  
        '''
        Gradient of objective function
        '''
        assert len(x) == int(NUM_MPC_INPUTS)
        step = 1.e-3 # we just have a small step
        objp=objm=0
        grad_f = [0]*NUM_MPC_INPUTS
        xpstep = [0]*NUM_MPC_INPUTS
        xmstep = [0]*NUM_MPC_INPUTS
        for i_mpc_input in range(NUM_MPC_INPUTS):
            xpstep=x.copy()
            xmstep=x.copy()
            # for each variables, we need to evaluate the derivative of the function with respect to that variable, This is why we have the for loop
            xpstep[i_mpc_input]  = xpstep[i_mpc_input]+step 
            xmstep[i_mpc_input] = xmstep[i_mpc_input]-step
            #print ("step: ", step)
            #print ("xp:  ",xpstep)
            #print ("xm:  ",xmstep)
            #print ("i_mpc_input:  ",i_mpc_input)
            # Evaluate the objective function at xpstep and xmstep
            objp=eval_f(xpstep) # This function returns the value of the objective function evaluated with the variable x[i] is perturebed +step
            objm=eval_f(xmstep) # This function returns the value of the objective function evaluated with the variable x[i] is perturebed -step
            #print ("obj ", objp, "   objm   ", objm)
            grad_f[i_mpc_input] = (objp - objm) / (2 * step) # This evaluates the gradient of the objetive function with repect to the optimization variable x[i]
        #print("Gradient: ", grad_f)
        return array(grad_f)
    def eval_g(x):  
        '''
        Define MPC constraints
        ''' 
        assert len(x) == int(NUM_MPC_INPUTS)
        CAd2=realtime_data[0]  ## current CA
        Td2=realtime_data[1]  ## current T
        g=array([-5.0]*NUM_MPC_CONSTRAINTS)   # g is the constraint (inequality) value; Initilize to be negative.
    

        if ((a*CAd2**2+d*Td2**2+2*b*CAd2*Td2-2)> 0):  ## If V > \rho_min=2
            LfV = (2 * a * CAd2 + 2 * b * Td2) * ((F / V) * (-CAd2) - k0 * ((numpy.exp(-E / (R * (Td2 + Ts))) * (CAd2 + CAs) ** 2)-
                                                                            numpy.exp(-E / (R * Ts)) *(CAs)**2)) + \
                (2 * d * Td2 + 2 * b * CAd2) * (((F / V) * (-Td2) + (-Dh / (sigma * cp)) *
                                                (k0 * ((numpy.exp(-E / (R * (Td2 + Ts))) *(CAd2 + CAs) ** 2) - numpy.exp(-E / (R * Ts)) * CAs ** 2))))

            #LfV= dV/dx * f

            LgV = (2 * d * Td2 + 2 * b * CAd2) / (V * sigma * cp)  #LgV= dV/dx * g
    
            h2x = -(LfV + sqrt((LfV ** 2) + LgV ** 4)) / (LgV)  # h2x= \phi_nn(x)


            if (h2x > 5e5):
                h2x=5e5

            if (h2x < -5e5):
                h2x=-5e5

            dot_Vt = (2 * a * CAd2 + 2 * b * Td2) * ((F / V) * (0 - CAd2) -
                                                    k0 * ((numpy.exp(-E / (R * (Td2 + Ts))) * (
                                CAd2 + CAs) ** 2) - numpy.exp(
                        -E / (R * Ts)) * (CAs ** 2))) + \
                    (2 * d * Td2 + 2 * b * CAd2) * (((F / V) * (-Td2) + (-Dh / (sigma * cp)) * (
                    k0 * ((numpy.exp(-E / (R * (Td2 + Ts))) * (CAd2 + CAs) ** 2) -
                        numpy.exp(-E / (R * Ts)) * CAs ** 2)) + (h2x / (sigma * cp * V))))

                    #dot_Vt= \dot V on RHS, \dot V= dV/dx * dx/dt

            dot_V=(2*a * CAd2 + 2*b * Td2)*((F / V)*(x[0] - CAd2) -
                k0*((numpy.exp(-E / (R*(Td2 + Ts)))* (CAd2 + CAs) ** 2) - numpy.exp(-E / (R*Ts))*(CAs ** 2))) + \
                (2*d*Td2 + 2*b * CAd2)*(((F / V)*(-Td2) + (-Dh / (sigma*cp))*(k0*((numpy.exp(-E / (R*(Td2 + Ts)))* (CAd2 + CAs)** 2) -
                numpy.exp(-E / (R*Ts))* CAs ** 2)) + (x[1] / (sigma*cp*V))))

                ## dot_V= \dot V on LHS
            #print ("dot V", dot_V)
            g[0]=dot_V-dot_Vt    # this corresponds to the constraint: dot_V (x,u) < dot_V(x, \phi(x))
            #g[0]=dot_V+10

        else:
            ### if V > \rho_min=2
            df_ensemble_output2 = my_ens_prediction(num_horizon = HORIZON, my_rawdata=realtime_data,
                                                my_inputs=x)
            

            #####  only account for the last point #####
            for j in range(HORIZON):
                est_outlet_product2 = df_ensemble_output2[j, -1, 0:2]
                # for i in range(NUM_IN_SEQUENCE):
                g[j]= a * est_outlet_product2[0] ** 2+ 2 * b * est_outlet_product2[1]*est_outlet_product2[0] + \
                    d*est_outlet_product2[1] ** 2 -1.8  # V=a*CA^2+ d*T^2 +2*b*CA*T
                # this corresponds to the constraint: V(x,u) < 1.8 (\rho_min or \rho_nn in different papers)
    
        return  g

    nnzj = NUM_MPC_CONSTRAINTS*NUM_MPC_INPUTS


    def eval_jac_g(x, flag):  
        '''
        Define JOCABIAN OF constraints
        '''
        #print ("in eval_jac_g_0")
        if flag:
            list_x = []
            list_y=[]
            for i in range(HORIZON):
                list_x = list_x + [i] * NUM_MPC_INPUTS
                list_y = list_y +list(range(0, int(NUM_MPC_INPUTS)))
            #list_x=[0]*int(NUM_MPC_INPUTS)+[1]*int(NUM_MPC_INPUTS)
            #list_y=list(range(0, int(NUM_MPC_INPUTS)))+list(range(0, int(NUM_MPC_INPUTS)))
            #print ("list_x:", list_x)
            #print("list_y:", list_y)
            return (array(list_x),
                    array(list_y))

            #return (array([0, 0]),
            #        array([0, 1]))
            #print ("in eval_jac_g_1")
        else:
            assert len(x) == int(NUM_MPC_INPUTS)
            step = 1e-3 # we just have a small step
            gp=gm=numpy.zeros(NUM_MPC_CONSTRAINTS)
            xpstep=xmstep=numpy.zeros(NUM_MPC_INPUTS)
            jac_g = [[0]*int(NUM_MPC_INPUTS) for _ in range(NUM_MPC_CONSTRAINTS)]
            #print ("shape:", jac_g)
            for i_mpc_input in range(NUM_MPC_INPUTS):
                xpstep=x.copy()
                xmstep=x.copy()
                # for each variables, we need to evaluate the derivative of the function with respect to that variable, This is why we have the for loop
                xpstep[i_mpc_input] += step 
                xmstep[i_mpc_input] -= step
                gp=eval_g(xpstep)
                gm=eval_g(xmstep)
                for num_constraint in range(NUM_MPC_CONSTRAINTS):
                    jac_g[num_constraint][i_mpc_input] = (gp[num_constraint] - gm[num_constraint]) / (2 * step)
                #print ("in eval_jac_g_2:")
            return array(jac_g)

    def apply_new(x):
        return True
    def print_variable(variable_name, value):
        for i in range(len(value)):
            print("{} {}".format(variable_name + "["+str(i)+"] =", value[i]))


    nnzh = NUM_MPC_INPUTS**2  ## non-zero elements in Hessian matrix

    # def eval_h(x, lagrange, obj_factor, flag, user_data = None):
    #     if flag:
    #         hrow = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    #         hcol = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]
    #         return (array(hcol), array(hrow))
    #     else:
    #         values = zeros((10), float_)
    #         values[0] = obj_factor * (2*x[3])
    #         values[1] = obj_factor * (x[3])
    #         values[2] = 0
    #         values[3] = obj_factor * (x[3])
    #         values[4] = 0
    #         values[5] = 0
    #         values[6] = obj_factor * (2*x[0] + x[1] + x[2])
    #         values[7] = obj_factor * (x[0])
    #         values[8] = obj_factor * (x[0])
    #         values[9] = 0
    #         values[1] += lagrange[0] * (x[2] * x[3])

    #         values[3] += lagrange[0] * (x[1] * x[3])
    #         values[4] += lagrange[0] * (x[0] * x[3])

    #         values[6] += lagrange[0] * (x[1] * x[2])
    #         values[7] += lagrange[0] * (x[0] * x[2])
    #         values[8] += lagrange[0] * (x[0] * x[1])
    #         values[0] += lagrange[1] * 2
    #         values[2] += lagrange[1] * 2
    #         values[5] += lagrange[1] * 2
    #         values[9] += lagrange[1] * 2
    #         return values

    dir_name = os.getcwd()
    test = os.listdir(dir_name)

    for item in test:
        if item.endswith(".txt"):
            os.remove(os.path.join(dir_name, item))


    nvar = NUM_MPC_INPUTS

    ##  DEFINE THE UPPER BOUND AND LOWER BOUND FOR optimized variables ###
    x_lower=[0]* nvar
    x_upper=[0]* nvar
    for i in range(int(NUM_MPC_INPUTS/2)):
        x_lower[2*i]= -3.5 
        x_lower[2 * i+1] = -5e5
        x_upper[2 * i] = 3.5
        x_upper[2 * i + 1] = 5e5
    x_L = array(x_lower) #array([-3.5,-5e5])
    x_U = array(x_upper) #array([3.5, 5e5])

    ### DEFINE THE UPPER BOUND AND LOWER BOUND OF THE CONSTRAINT ###
    ncon = NUM_MPC_CONSTRAINTS
    g_L = array([-2e19]*HORIZON)
    g_U = array([0]*HORIZON)

    print ("g_L", g_L, g_U)

    #####################################################################
    ##### PRE-PROCESSING (THE FOLLOWING COMMANDS ARE EXECUTED ONCE) #####
    #####################################################################
    #### LOAD MEAN AND STD FILES###########
    #### READ MEANS & STD FROM THE FILE #####
        
        
    ####################################################################
    ##### SOLVING THE MPC PROGRAM TO FIND THE OPTIMIZED MPC INPUTS #####
    ####################################################################
    ##########  KEEP RUNNING MPC ###############

    rawdata=numpy.array([CAi,Ti])  ## global variable

    # Introduce a timer to keep track of the operation time
    global current_time
    current_time = 0.0
    
    for main_iteration in range(NUM_MPC_ITERATION):
        # print ("Num Iteratin: ", main_iteration)
    

        #### NORMALIZE RAW DATA ####
        #rawdata=(rawdata-state_mean)/state_std
        # print ("normalized data:  ", rawdata)

        realtime_data=rawdata   ## current state variables global variable

        nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g)
        #x0 = array([3.33, 3.8, 3.8, 6])


        ### INITIAL GUESS of manipulated variables X ###
        if main_iteration ==0 :
            x0 = array([0.0]*int(NUM_MPC_INPUTS))
        else:
            x0=x
            x0[0:-2]= x[2:]
            x0[-2:]=[0, 0]
            # x0[-2:]= x[-2:]
            #x0=array(x[:-2], 0, 0)
        """
        print x0
        print nvar, ncon, nnzj
        print x_L,  x_U
        print g_L, g_U
        print eval_f(x0)
        print eval_grad_f(x0)
        print eval_g(x0)
        a =  eval_jac_g(x0, True)
        print "a = ", a[1], a[0]
        print eval_jac_g(x0, False)
        print eval_h(x0, pi0, 1.0, False)
        print eval_h(x0, pi0, 1.0, True)
        """

        """ You can set Ipopt options by calling nlp.num_option, nlp.str_option
        or nlp.int_option. For instance, to set the tolarance by calling

            nlp.num_option('tol', 1e-8)

        For a complete list of Ipopt options, refer to

            http://www.coin-or.org/Ipopt/documentation/node59.html

        Note that Ipopt distinguishs between Int, Num, and Str options, yet sometimes
        does not explicitly tell you which option is which.  If you are not sure about
        the option's type, just try it in PyIpopt.  If you try to set one type of
        option using the wrong function, Pyipopt will remind you of it. """

        nlp.int_option('max_iter', 500)
        nlp.num_option('tol', 1e-5)
        nlp.int_option('print_level', 2)
        # print("Going to call solve")
        # print("x0 = {}".format(x0))
        x, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)

        # import pdb; pdb.set_trace()
        nlp.close()

        # print("Solution of the primal variables, x")
        # print_variable("x", x)
        # print("status = ", status)

        # print("Solution of the bound multipliers, z_L and z_U")
        # print_variable("z_L", zl)
        # print_variable("z_U", zu)

        # print("Solution of the constraint multipliers, lambda")
        # print_variable("lambda", constraint_multipliers)

        # print("Objective value")
        # print("f(x*) = {}".format(obj))
        # print ("Control action=:  ", x[0], x[1])

        #REAL_CONTROL_ACTION=x*
        #print ("REAL DATA:", REAL_CONTROL_ACTION)
        #numpy.savetxt("input_to_fluent.out",   REAL_CONTROL_ACTION, fmt="%f",  delimiter=" ")
        
        ###################################### Apply optimal control actions u1 and u2 to real system   #########################################
        ###################################### Apply only the first horizon control action to real system   #####################################
        x1=CAi
        x2=Ti

        for kk in range (int(delta/hc)):
            #print ("Anh needs to print x1 and x2: ", x1, x2)
            #x1_new = x1 + hc * ((F / V) * (u1  - x1) - k0 * (numpy.exp(-E / (R * x2))*x1 * x1))
            #x2_new = x2 + hc * ((F / V) * (T0-x2) + (-Dh / (sigma * cp)) * k0 * (numpy.exp(-E / (R * x2)) * x1* x1)
            #        + (u2 / (sigma * cp * V)))


            x1_new = x1 + hc * ((F / V) * (x[0] - x1) -
                                k0 * ((numpy.exp(-E / (R * (x2 + Ts)))*(x1 + CAs) * (x1 + CAs))
                                    - numpy.exp(-E / (R * Ts)) * CAs * CAs))

            x2_new = x2 + hc * (((F / V) * (-x2) + (-Dh / (sigma * cp)) *
                                (k0 * ((numpy.exp(-E / (R * (x2 + Ts))) * (x1 + CAs) * (x1 + CAs)) -
                                        numpy.exp(-E / (R * Ts)) * CAs * CAs)) + (x[1] / (sigma * cp * V))))

            x1 = x1_new
            x2 = x2_new
            
        # updated the state variables for next sampling time
        CAi=x1
        Ti=x2

            #print ("First principle intermediate steps x1, x2:  ", x1, x2)
            #print("First principle intermediate derivatives x1, x2:  ", x1_derivate, x2_derivate)

        #x1=x1-CAs
        #x2=x2-Ts
        # print('Real model output x1 x2 in deviation form:   ', x1, x2)
        #x1=(x1 -y1_mean)/y1_std
        #x2 = (x2 - y2_mean) / y2_std
        
         # Surge Attack
        # Assume: attack occur at i0 = 0.0010 hr, 
                # duration of the initial surge, Ls = 0.0010, 
                # total duration of the attack, La = 0.01 

    # when 0.001 < t < 0.002, x(ti) = min /max {x|V(x(ti)) = 372}
        if current_time >= 0.001 and current_time < 0.002:
            
            coefficients = [P[1,1], 2*P[1,0]*CAi, (P[0,0]*CAi*CAi-372)]
            roots = np.roots(coefficients)
            Ti_attack = min(roots[0],roots[1])

    # when 0.002 < t < 0.01, x(ti) = x(ti) + n(ti)
        elif current_time >= 0.002 and current_time <= 0.01:
            noise = np.random.normal(0, 1)  # noise with mean=0 and std=1
            noise = max(-2, min(noise, 2))  # Bound the noise between [-2, 2]
            Ti_attack = Ti + noise
                
            Vx = P[1, 1] * Ti_attack * Ti_attack + 2 * P[1, 0] * CAi * Ti_attack + P[0, 0] * CAi * CAi
            
          
        if Vx >= 372:
            coefficients = [P[1,1], 2*P[1,0]*CAi, (P[0,0]*CAi*CAi-372)]
            roots = np.roots(coefficients)
            Ti_attack = min(roots[0],roots[1])
        else:
                Ti_attack = Ti_attack

        rawdata = numpy.array([CAi, Ti_attack]) 
          

        # print ("x2_sensor_record: ",rawdata[1]) 
        x2_sensor_record.append(rawdata[1])
        
        x1_record.append(x1)
        x2_record.append(x2) 
        u1_record.append(x[0])
        u2_record.append(x[1])
        current_time += hc  # Update timer by hc

    # print ("x1_record: ",x1_record)
    # print ("x2_record: ",x2_record)
    # print ("x2_sensor_record: ",x2_sensor_record)
    # print ("u1_record: ",u1_record)
    # print ("u2_record: ",u2_record)

    return x1_record[:-1], x2_record[:-1], x2_sensor_record[:-1], u1_record, u2_record

T_initial = np.linspace(300, 600, 30, endpoint=True) - Ts
CA_initial = np.linspace(0, 6, 30, endpoint=True) - CAs

# sieve out initial states that lie outside of stability region

T_start = list()
CA_start = list()

for T in T_initial:
    for CA in CA_initial:
        x = np.array([CA, T])
        if x @ P @ x < 372:
            CA_start.append(CA)
            T_start.append(T)
print("number of initial conditions: {}".format(len(CA_start)))

# convert to np.arrays
CA_start = np.array([CA_start])
T_start = np.array([T_start])
x_deviation = np.concatenate((CA_start.T, T_start.T), axis=1)  # every row is a pair of initial states within stability region
print("shape of x_deviation is {}".format(x_deviation.shape))
CA_input = list()
T_input = list()
T_sensor_input = list()
CA0_input = list()
Q_input = list()
for C_A_initial, T_initial in x_deviation:
    CA_record, T_record, T_sensor_record, CA0_record, Q_record =\
        MPC_SIMULATION(C_A_initial, T_initial)
    
    CA_input.append(CA_record)
    T_input.append(T_record)
    T_sensor_input.append(T_sensor_record)
    CA0_input.append(CA0_record)
    Q_input.append(Q_record)

# collate input for RNN

CA_input = np.array(CA_input)
CA_input = CA_input.reshape(-1, 10, 1)

T_input = np.array(T_input)
T_input = T_input.reshape(-1, 10, 1)

T_sensor_input = np.array(T_sensor_input)
T_sensor_input = T_sensor_input.reshape(-1, 10, 1)

CA0_input = np.array(CA0_input)
CA0_input = CA0_input.reshape(-1, 10, 1)

Q_input = np.array(Q_input)
Q_input = Q_input.reshape(-1, 10, 1)

RNN_input = np.concatenate((CA_input, T_sensor_input, CA0_input, Q_input), axis=2)
print("RNN_input shape is {}".format(RNN_input.shape))  # output shape: number of samples x timestep x variables

# collate output for RNN

RNN_output = np.zeros((68, 1)) #change this number to the number of samples
# checking output

print('RNN_output shape is',RNN_output.shape)
np.save("RNN_input_attack.npy", RNN_input)
np.save("RNN_output_attack.npy", RNN_output)
np.save('T_input_attack.npy', T_input)
