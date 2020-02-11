from lav_ess.linmod import *
from lav_ess.main import *
from lav_ess.system import System
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lav_ess.gurobinate
import gurobipy
from gurobipy import *
import logging
logger = logging.getLogger(__name__)


def PV_plant(name='Utility', P_PV=1e6, price=1):
    plant = Asset(name=name)
    P_PV_out = plant.add('P_PV_out',  lower_bound=P_PV, upper_bound=P_PV,outflow=True)
    price = plant.add('price', value=price)
    plant.add('income', expr=price*P_PV_out,negative=True)
    return plant

def demand(name='demand', P_demand_max=1,price_tariff=1):
    node = Asset(name=name)
    P_demand = node.add('P_demand', lower_bound=P_demand_max, upper_bound=P_demand_max, outflow=False)
    price_tariff = node.add('price_tariff', value=price_tariff)
    node.add('income', expr=price_tariff*P_demand,negative=True)
    return node

def importPower (name='Import', P_import_max=200, price_imp=1):
    node = Asset(name=name)
    P_import = node.add('P_import', lower_bound=0, upper_bound=P_import_max, outflow=True)
    price_imp = node.add('price_import', value=price_imp)
    node.add('cost', expr=price_imp*P_import,negative=True)
    return node

def exportPower (name='Export', P_export_max=200):
    node = Asset(name=name)
    P_export = node.add('P_export', lower_bound=0, upper_bound=P_export_max,outflow=False)
    return node

def battery (name='Battery', P_batt_charge=1, P_batt_discharge=1, SOC_max=1, SOC_min=1,eta_charge=1, eta_discharge=1):
    node = Asset(name=name)
    Batt_charge = node.add('Batt_charge', lower_bound=0, upper_bound=P_batt_charge,outflow=False)
    Batt_discharge=node.add('Batt_discharge', lower_bound=0, upper_bound=P_batt_discharge, outflow=True)
    Batt_SOC = node.add('Batt_SOC', lower_bound=SOC_min, upper_bound=SOC_max, 
                         periodicity=True)
    node.add(Batt_SOC.diff() == Batt_charge*eta_charge-Batt_discharge*eta_discharge)
    
    return node


def methanation (name='Methanizer', SOC_max_PtG=1 ,SOC_min_PtG=1, P_PtG=1, P_elec=1, P_batt_PtG_charge=1, 
                 P_batt_PtG_discharge=1, eta_meth=0.65, eta_elec = 0.8, eta_charge_PtG=1,eta_discharge_PtG=1,
                 price_methan=80,tau_elec=1,tau_PtG=1):
    node = Asset(name=name)
    P_CH4=node.add('P_CH4', lower_bound=0, upper_bound=P_PtG )
    P_electrolizer = node.add('P_electrolizer', lower_bound=0, upper_bound=P_elec )
    eta_meth = node.add('eta_meth', value=eta_meth)
    eta_elec = node.add('eta_elec', value=eta_elec)
    price_methan = node.add('price_methan', value=price_methan)
    Batt_PtG_charge = node.add('Batt_PtG_charge', lower_bound=0, upper_bound=P_batt_PtG_charge)
    Batt_PtG_discharge=node.add('Batt_PtG_discharge', lower_bound=0, upper_bound=P_batt_PtG_discharge)
    Batt_PtG_SOC = node.add('Batt_PtG_SOC', lower_bound=SOC_min_PtG, upper_bound=SOC_max_PtG)#,periodicity=True)
    node.add(Batt_PtG_SOC.diff() == Batt_PtG_charge*eta_charge_PtG-Batt_PtG_discharge*eta_discharge_PtG)
    node.add(Batt_PtG_discharge == P_electrolizer*eta_elec)
    node.add(P_electrolizer == P_CH4*eta_meth)
    node.add('benefit', expr=P_CH4*price_methan,negative=True)
    tau_elec = node.add('tau_elec', value=tau_elec)
    tau_PtG = node.add('tau_PtG', value=tau_PtG)
    delta_PtG = node.add('delta_PtG', lower_bound=-P_PtG/tau_PtG, upper_bound=P_PtG/tau_PtG, negative=True)
    delta_elec = node.add('delta_elec', lower_bound=-P_elec/tau_elec, upper_bound=P_elec/tau_elec, negative=True)
    node.add(P_CH4.diff() == delta_PtG)
    node.add(P_electrolizer.diff() == delta_elec)
    
    return node


def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc(text):
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time for **"+ text +"** " + str(round((time.time() - startTime_for_tictoc),4)) + " s")
    else:
        print ("Toc: start time not set")

def make_europe():
    sys = System('EU')
    dfi = pd.read_csv('data_input.csv')
    nhours = 8760
    #Prices definition
    price_market = np.zeros(nhours)
    price_market[:] = dfi['real price 2015'][:nhours]
    
    price_tarif = np.zeros(nhours)  # euro/MWh
    price_tarif[:] = 100 # this is just a guess for now
    
    # Price contracting with PV from utility (smaller than regular tarif)
    price_PV = np.zeros(nhours)  # euro/MWh
    price_PV[:] = 80 # this is just a guess for now
    
    price_gas = np.ones(nhours)*100  # euro/MWh
    cost_CO2_tonne_PtG = 100 # [CHF/tonneCO2]
    # 1 tonne CO2 corresponds to 5 MWh_CH4
    cost_CO2_MWh_PtG = cost_CO2_tonne_PtG / 5.0 # [CHF/MWh_CH4]

    demand1 = np.zeros(nhours)
    demand1[:] = dfi['Demand CH'][:nhours]
    # Scale-down the demand from Switzerland (60 TWh) to region (1.1 TWh)
    demand1 = demand1 * 1.1e6 / np.sum(demand1)
    # PV installations are distinguished between utility and private
    # Private doesn't enter as sell for utility
    IC_PV_utility = 20
    IC_PV_private = 50 # Installed capacity PV [MW]
    # Normalize production is the same for both categories
    PV_norm = dfi['CF_solar CH'][:nhours] # normalized PV production
    PV_prod_utility = PV_norm * IC_PV_utility
    PV_prod_private = PV_norm * IC_PV_private
    P_import_max1 = np.zeros(nhours)
    P_import_max1[:] = 200  
    P_export_max1 = np.zeros(nhours)
    P_export_max1[:] = 200 
    
    P_batt_charge_in = 20 # Installed power battery [MW]
    P_batt_discharge_in = 20 # Installed power battery [MW]
    eta_charge1 = 0.9
    eta_discharge1 = 0.95
    Cap_batt = 500 # Capacity battery [MWh]
    SOC_min_in = 0.1 * Cap_batt
    SOC_max_in = 1.0 * Cap_batt
    
    # Power to gas plant
    # As initial step gas is generated and fed into the gas grid
    # with a remuneration price of the gas market price
    P_PtG1 = 1 # Installed power in terms of CH4 heating value [MW]
    P_elec1 = 2 * P_PtG1 # Installed power electrolizer [MW]
    P_batt_PtG1 = 10 * P_PtG1  # Power auxiliary battery [MW]
    # Daily battery cycle to redistribute solar peaks along 24 h
    Cap_batt_PtG = 60 * P_PtG1  # Capacity auxiliary battery [MWh]
    SOC_min_PtG1 = 0.1 * Cap_batt_PtG
    SOC_max_PtG1 = 1.0 * Cap_batt_PtG  
    eta_meth1=0.65
    eta_elec1=0.8
    eta_charge_PtG1 = 0.9
    eta_discharge_PtG1 = 0.95
    
    # Speed ramps expressed in inv(dP/dt) -> time needed to switch to full load
    tau_PtG1 = 100 # characteristic time methanizer [hours]
    tau_elec1 = 1 # characteristic time electrolizer [hours]
    tau_batt = 0 # characteristic time aux battery [hours]
    
    # Prices used in the model
    price_tariff_PV_util=(price_PV-price_tarif)
    price_tariff_PV_priv=(-price_tarif)
    price_tariff_demand=(price_tarif)
    price_import=(-price_market)
    price_methane=(price_gas-cost_CO2_MWh_PtG)

    country = System(name='CH')
    grid = country.add(Connection('grid'))
    
    p_PV_util = country.add(PV_plant(name='Utility', P_PV=PV_prod_utility, price=price_tariff_PV_util))
    grid.connect(p_PV_util['P_PV_out'])
    
    p_PV_priv = country.add(PV_plant(name='Private', P_PV=PV_prod_private, price=price_tariff_PV_priv))
    grid.connect(p_PV_priv['P_PV_out'])
    
    d1 = country.add(demand(P_demand_max=demand1,price_tariff=price_tariff_demand))
    grid.connect(d1['P_demand'])
    
    P_import = country.add(importPower(P_import_max=P_import_max1, price_imp=price_import))
    grid.connect(P_import['P_import'])
    
    P_export = country.add(exportPower(P_export_max=P_export_max1))
    grid.connect(P_export['P_export'])
    
    batt=country.add(battery(name='Battery', P_batt_charge=P_batt_charge_in, P_batt_discharge=P_batt_discharge_in, 
                             SOC_max=SOC_max_in, SOC_min=SOC_min_in, eta_charge=eta_charge1, eta_discharge=1/eta_discharge1))
    grid.connect(batt['Batt_charge'])
    grid.connect(batt['Batt_discharge'])

    meth=country.add(methanation (name='Methanizer', SOC_max_PtG=SOC_max_PtG1 ,SOC_min_PtG=SOC_min_PtG1,
                                  P_PtG=P_PtG1, P_elec=P_elec1, P_batt_PtG_charge=P_batt_PtG1,
                                  P_batt_PtG_discharge=P_batt_PtG1,eta_meth=1/eta_meth1,
                                  eta_elec = 1/eta_elec1, eta_charge_PtG=eta_charge_PtG1,eta_discharge_PtG=1/eta_discharge_PtG1,
                                  price_methan=price_methane,tau_elec=tau_elec1,tau_PtG=tau_PtG1))
    grid.connect(meth['Batt_PtG_charge'])
    sys.add(country)

    #pl = Line.connect(sys['CH']['grid'], sys['FR']['grid'])
    #sys.add(pl)

    return sys


#def main():
sys = make_europe()
model = gurobipy.Model()
logging.basicConfig(level=logging.DEBUG)
logger.debug('creating Gurobi variables')
g = lav_ess.gurobinate.Gurobinator(sys, model=model)
g.process(sys)
model.update()
op = Operator()
op.connect(sys['CH']['Utility']['income'])
op.connect(sys['CH']['Private']['income'])
op.connect(sys['CH']['demand']['income'])
op.connect(sys['CH']['Import']['cost'])
op.connect(sys['CH']['Methanizer']['benefit'])
model.setObjective((g.integrate(op.sum())),sense=GRB.MAXIMIZE)
model.update()
model.optimize()

toc('End of calculations')    
print('Obj: %g' % model.objVal)
g.x.copy_solution(model)
    
#    plt.plot(sys['CH']['Private']['P_PV_priv_out'].solution,label='PVpriv',color='cyan')
#    plt.plot(sys['CH']['Utility']['P_PV_util_out'].solution,label='PVutil',color='blue')
#    plt.plot(sys['CH']['demand']['P_demand'].solution,label='Demand',color='red')
#    plt.plot(sys['CH']['Import']['P_import'].solution,label='Import',color='green')
#    plt.plot(sys['CH']['Export']['P_export'].solution,label='Export',color='yellow')
#    plt.legend()
#    plt.show()
#    
#    plt.plot(sys['CH']['Methanizer']['P_electrolizer'].solution,label='H2',color='green')
#    plt.plot(sys['CH']['Methanizer']['P_CH4'].solution,label='CH4',color='blue')
#    plt.legend()
#    plt.show()
#    
#    plt.xlim((20,300))
#    plt.plot(sys['CH']['Battery']['Batt_charge'].solution,label='Charge',color='red')
#    plt.plot(sys['CH']['Battery']['Batt_discharge'].solution,label='Discharge',color='green')
#    plt.legend()
#    plt.show()  
#    
##    plt.xlim((20,300))
#    plt.plot(sys['CH']['Methanizer']['Batt_PtG_charge'].solution,label='Charge_PtG',color='red')
#    plt.plot(sys['CH']['Methanizer']['Batt_PtG_discharge'].solution,label='Disharge_PtG',color='green')
#    plt.legend()
#    plt.show() 
#    
#    plt.plot(sys['CH']['Utility']['income_Util'].solution,label='Income_util')
#    plt.plot(sys['CH']['Private']['income_Priv'].solution,label='Income_priv')
#    plt.plot(sys['CH']['demand']['income'].solution,label='Income_demand')
#    plt.plot(sys['CH']['Import']['cost'].solution,label='Cost_import')
#    plt.plot(sys['CH']['Methanizer']['benefit'].solution,label='Benefit_methanizer')
#    plt.legend()
#    plt.show() 
    
#    import code
#    code.interact(local=locals())
    



#if __name__ == '__main__':
#    tic()
#    logging.basicConfig(level=logging.DEBUG)
#    main()
#    