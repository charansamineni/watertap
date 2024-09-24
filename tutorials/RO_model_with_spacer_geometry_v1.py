# Written by Charan Samineni
# Importing required modules
import matplotlib.pyplot as plt
from pyomo.environ import (Param, Var, Constraint, TransformationFactory, ConcreteModel,
                           value, assert_optimal_termination, Reals, units as pyunits)
from pyomo.network import Arc
# Idaes core components
from idaes.core import FlowsheetBlock
from idaes.core.util.scaling import calculate_scaling_factors, set_scaling_factor
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver
from idaes.core.util.scaling import constraint_scaling_transform
from idaes.core.util.initialization import propagate_state
from idaes.models.unit_models import Feed, Product
# WaterTAP components
import watertap.property_models.seawater_prop_pack as prop
from watertap.unit_models.reverse_osmosis_1D import (
    ReverseOsmosis1D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType
)
from watertap.unit_models.pressure_changer import Pump


def build():
    # Build a concrete model and create a flowsheet
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # Add seawater prop pack to the flow sheet
    m.fs.properties = prop.SeawaterParameterBlock()

    # Build necessary components for the flow sheet
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.product = Product(property_package=m.fs.properties)
    m.fs.pump = Pump(property_package=m.fs.properties)
    m.fs.RO = ReverseOsmosis1D(
        property_package=m.fs.properties,
        has_pressure_change=True,
        pressure_change_type=PressureChangeType.calculated,
        mass_transfer_coefficient=MassTransferCoefficient.calculated,
        concentration_polarization_type=ConcentrationPolarizationType.calculated,
    )

    # Connect the elements of the Flow sheet

    m.fs.feed_to_pump = Arc(source=m.fs.feed.outlet, destination=m.fs.pump.inlet)
    m.fs.pump_to_ro = Arc(source=m.fs.pump.outlet, destination=m.fs.RO.inlet)
    m.fs.ro_to_product = Arc(source=m.fs.RO.permeate, destination=m.fs.product.inlet)

    # Expand the arcs
    TransformationFactory("network.expand_arcs").apply_to(m)
    return m


def scaling(m):
    # Fix the variables and scaling factors for Feed

    # m.fs.feed.properties[0].display()
    m.fs.feed.properties[0].temperature.fix(293.15)
    m.fs.feed.properties[0].pressure.fix(101325)
    m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'H2O'].fix(0.965)
    m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'TDS'].fix(0.035)

    # Construct the concentration values
    m.fs.feed.properties[0].conc_mass_phase_comp[...]
    # m.fs.feed.properties[0].display()

    # Scaling factor definition
    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1 / 0.965, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1 / 0.035, index=("Liq", "TDS"))

    m.fs.feed.properties[0].pressure_osm_phase[...]

    # Define the deafult values and scaling for pump
    m.fs.pump.efficiency_pump[0].fix(0.70)
    set_scaling_factor(m.fs.pump.control_volume.work, 1e-4)
    set_scaling_factor(m.fs.pump.control_volume.properties_out[0].pressure, 1e-5)
    set_scaling_factor(m.fs.pump.control_volume.properties_in[0].pressure, 1e-5)

    # RO membrane definition and scaling factors
    m.fs.RO.feed_side.velocity[0, 0].fix(0.1)
    m.fs.RO.area.fix(50)
    set_scaling_factor(m.fs.RO.area, 1 / 50)
    m.fs.RO.length.unfix()
    set_scaling_factor(m.fs.RO.length, 0.1)
    m.fs.RO.width.unfix()
    set_scaling_factor(m.fs.RO.width, 0.1)
    m.fs.RO.permeate.pressure[0].fix(101325)
    m.fs.RO.feed_side.channel_height.fix(1e-3)
    m.fs.RO.feed_side.spacer_porosity.fix(0.9)
    m.fs.RO.A_comp[0, "H2O"].fix(3 / (3600 * 1000 * 1e5))
    m.fs.RO.B_comp[0, "TDS"].fix(0.15 / (3600 * 1000))
    calculate_scaling_factors(m)


def Init(m):
    # Initialize the feed, pump, and RO unit
    solver = get_solver()
    m.fs.feed.initialize(optarg=solver.options)
    propagate_state(m.fs.feed_to_pump)
    # Get the pump pressure initialzied based on the osmotic pressure - RO block does not solve if this step is missing
    osmotic_feed_pressure = value(m.fs.feed.properties[0].pressure_osm_phase['Liq'])
    print("Osmotic pressure on the feed side is {} bar".format(osmotic_feed_pressure / 1e5))
    m.fs.pump.outlet.pressure[0].fix(osmotic_feed_pressure * 1.5)
    m.fs.pump.initialize(optarg=solver.options)
    propagate_state(m.fs.pump_to_ro)
    m.fs.RO.initialize(optarg=solver.options)


def solve(m):
    # Solve a simple problem at the initialized state
    print("The DOF are {} and the expected value is 0".format(degrees_of_freedom(m)))
    assert degrees_of_freedom(m) == 0
    solver = get_solver()
    result = solver.solve(m, tee=True)
    assert_optimal_termination(result)

    m.fs.pump.outlet.pressure.unfix()
    m.fs.RO.recovery_vol_phase[0, 'Liq'].fix(0.50)
    assert degrees_of_freedom(m) == 0
    result = solver.solve(m, tee=False)
    assert_optimal_termination(result)
    # m.fs.RO.report()


def salvsP(m):
    solver = get_solver()
    m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'TDS'].unfix()
    # Create the concentrations you are interested in
    import numpy as np
    concentrations = np.linspace(10, 90, 5)
    OperPressures = []
    for c in concentrations:
        m.fs.feed.properties[0].conc_mass_phase_comp['Liq', 'TDS'].fix(c)
        assert degrees_of_freedom(m) == 0
        result = solver.solve(m, tee=False)
        assert_optimal_termination(result)
        OperPressures.append(m.fs.RO.inlet.pressure[0].value * 1e-5)
    import matplotlib.pyplot as plot
    fig, ax = plot.subplots()
    ax.plot(concentrations, OperPressures)
    ax.set(xlabel='Concentration (g/L)', ylabel='Operating Pressure (bar)')
    # plot.show()


def Amodifications(m):
    import idaes.core.util.math as idaesMath
    import numpy as np
    # Import the necessary module for constructing the variable from a constraint
    from pyomo.util.calc_var_value import calculate_variable_from_constraint
    from idaes.core.util.model_statistics import report_statistics

    # Define new variable for A and set scaling factor
    m.fs.A_init = Var(initialize=2)
    m.fs.A_init.fix()
    set_scaling_factor(m.fs.A_init, 1 / m.fs.A_init.value)
    # Define a constraint that relates A = f(P)
    m.fs.RO.A_pressure_constraint = (
        Constraint(expr=m.fs.RO.A_comp[0, "H2O"] * (3600 * 1000 * 1e5) ==
                        idaesMath.smooth_min(m.fs.A_init,
                                             (m.fs.A_init * (65 * 1e5 / m.fs.RO.inlet.pressure[0])))))
    m.fs.RO.A_comp[0, 'H2O'].unfix()

    # Calculate the A values at a range of P values
    A_p = []
    pressure_values = np.linspace(10, 200, 5)
    for p in pressure_values:
        m.fs.RO.inlet.pressure[0] = p * 1e5
        calculate_variable_from_constraint(m.fs.RO.A_comp[0, 'H2O'], m.fs.RO.A_pressure_constraint)
        A_p.append(m.fs.RO.A_comp[0, 'H2O'].value * (3.6 * 1e11))

    # Plot the A values as a function of pressure
    import matplotlib.pyplot as plot
    fig, ax = plot.subplots()
    ax.plot(pressure_values, A_p)
    ax.set(xlabel='Pressure (bar)', ylabel='A_Comp (LMH/bar)')
    #plot.show()

    # Initialize A value at the operating pressure
    m.fs.RO.inlet.pressure[0] = m.fs.pump.outlet.pressure[0].value
    calculate_variable_from_constraint(m.fs.RO.A_comp[0, 'H2O'], m.fs.RO.A_pressure_constraint)

    print('DOF of the system are {}'.format(degrees_of_freedom(m)))
    assert degrees_of_freedom(m) == 0
    # m.fs.feed.properties[0].display()
    solver = get_solver()
    result = solver.solve(m, tee=False)
    assert_optimal_termination(result)

    # Calculate the Operating pressures with the change in A values
    c_values = np.linspace(10, 100, 10)
    p_with_adj = []
    A_with_adj = []
    for c in c_values:
        m.fs.feed.properties[0].conc_mass_phase_comp['Liq', 'TDS'].fix(c)
        print('DOF of the system are {}'.format(degrees_of_freedom(m)))
        # report_statistics(m)`
        result = solver.solve(m, tee=False)
        assert_optimal_termination(result)
        p_with_adj.append(m.fs.RO.inlet.pressure[0].value/1e5)
        A_with_adj.append(m.fs.RO.A_comp[0, 'H2O'].value*(3600*1000*1e5))
        print('Solved for a concentration of {}'.format(m.fs.feed.properties[0].conc_mass_phase_comp['Liq', 'TDS'].value))
        print('Operating pressure is {}'.format(m.fs.RO.inlet.pressure[0].value))

    # Generate a plot of Operating pressure dependence on A values
    fig, ax = plt.subplots()
    ax.plot(c_values, p_with_adj, color='blue', label='A = f(P)')
    ax.set(xlabel='Concentration (g/L)', ylabel='A value (LMH/bar)')
    plot.show()

if __name__ == '__main__':
    m = build()
    scaling(m)
    Init(m)
    solve(m)
    salvsP(m)
    Amodifications(m)
