# Importing necessary modules
import pyomo.core
from pyomo.core import ConcreteModel
from pyomo.environ import (Var, Param, Constraint, TransformationFactory,
                           ConcreteModel, value, assert_optimal_termination,
                           units as pyunits)
from pyomo.network import Arc
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from idaes.core import FlowsheetBlock
from idaes.core.util.scaling import calculate_scaling_factors, set_scaling_factor
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver
from idaes.core.util.initialization import propagate_state
from idaes.models.unit_models import Feed, Product
import watertap.property_models.seawater_prop_pack as prop
from watertap.unit_models.reverse_osmosis_1D import (
    ReverseOsmosis1D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType
)
from watertap.unit_models.pressure_changer import Pump
from idaes.core.util import DiagnosticsToolbox
import numpy as np
from pyomo.util.calc_var_value import calculate_variable_from_constraint
import matplotlib.pyplot as plot
from pyomo.util.infeasible import log_infeasible_constraints
import logging
from pyomo.util.check_units import assert_units_consistent


def main():
    model = build()
    scale_and_fix_flowsheet(model)
    initialize_model(model)
    add_spacer_porosity_correlation(model)
    verify_spacer_constraint(model, 20, 140, 13)
    # This step is necessary to make sure the flow sheet will reach optimal solution
    # when the solve_with_DaCosta_correlation routine is used
    add_mass_transfer_correlation_DaCosta(model)
    solve_with_DaCosta_mass_transfer_correlation(model, 20, 140, 13)
    # add_pressure_constraint_DaCosta(model)
    # check_pressure_constraint(model)
    # solve_with_DaCosta_dP_equation(model, 20, 140, 13)
    # report_results(model)
    plot.show(block=None)
    return model


def report_results(m):
    m.fs.pump.report()
    m.fs.RO.report()


def build():
    # Build the Flow sheet with a Concrete Model
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = prop.SeawaterParameterBlock()
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
    # Connect the Flow sheet
    m.fs.feed_to_pump = Arc(source=m.fs.feed.outlet, destination=m.fs.pump.inlet)
    m.fs.pump_to_RO = Arc(source=m.fs.pump.outlet, destination=m.fs.RO.inlet)
    m.fs.RO_to_product = Arc(source=m.fs.RO.permeate, destination=m.fs.product.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)
    return m


def scale_and_fix_flowsheet(m):
    # Setting the Scaling factors
    # Remove to display the state before fixing the variables m.fs.feed.display()
    m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'H2O'].fix(0.965)
    m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'TDS'].fix(0.035)
    m.fs.feed.properties[0].temperature.fix(298)
    m.fs.feed.properties[0].pressure.fix(101325)
    set_scaling_factor(m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'H2O'], 1 / 0.965)
    set_scaling_factor(m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'TDS'], 1 / 0.035)
    set_scaling_factor(m.fs.feed.properties[0].pressure, 1e-5)
    set_scaling_factor(m.fs.feed.properties[0].temperature, 1 / 300)

    m.fs.feed.properties[0].conc_mass_phase_comp[...]
    m.fs.feed.properties[0].pressure_osm_phase[...]

    # Fix Pump Variables and Scaling factors for pump unit model -
    # Note that Outlet pressure is not fixed here
    m.fs.pump.efficiency_pump.fix(0.7)
    set_scaling_factor(m.fs.pump.control_volume.properties_in[0].pressure, 1e-5)
    set_scaling_factor(m.fs.pump.control_volume.properties_out[0].pressure, 1e-5)
    set_scaling_factor(m.fs.pump.control_volume.work, 1e-4)

    # Fix the variables and Scaling factors for RO unit
    m.fs.RO.permeate.pressure[0].fix(101325)  # Permeate pressure fixed to atmospheric
    set_scaling_factor(m.fs.RO.permeate.pressure[0], 1e-5)
    m.fs.RO.A_comp[0, 'H2O'].fix(3 / (3600 * 1000 * 1e5))
    m.fs.RO.B_comp[0, 'TDS'].fix(0.15 / (3600 * 1000))
    m.fs.RO.area.fix(50)
    set_scaling_factor(m.fs.RO.area, 0.02)
    m.fs.RO.feed_side.spacer_porosity.fix(0.8)  # Spacer porosity values < 0.5 are creating
    # problems with initialization
    m.fs.RO.feed_side.channel_height.fix(0.001)
    set_scaling_factor(m.fs.RO.feed_side.channel_height, 1e3)
    m.fs.RO.feed_side.velocity[0, 0].fix(0.1)
    m.fs.RO.length.unfix()
    m.fs.RO.width.unfix()

    set_scaling_factor(m.fs.RO.permeate.pressure[0], 1e-5)
    set_scaling_factor(m.fs.RO.A_comp[0, 'H2O'], 1e10)
    set_scaling_factor(m.fs.RO.B_comp[0, 'TDS'], 1e5)
    set_scaling_factor(m.fs.RO.length, 0.1)
    set_scaling_factor(m.fs.RO.width, 0.1)

    # Setting scaling factors for all the mass concentration values at once
    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1 / 0.965, index=('Liq', 'H2O'))
    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1 / 0.035, index=('Liq', 'TDS'))

    calculate_scaling_factors(m)  # Calculate the remaining scaling factors if any


def initialize_model(m):
    # Initialize the model
    m.fs.feed.initialize()
    propagate_state(m.fs.feed_to_pump)
    osmotic_pressure = value(m.fs.feed.properties[0].pressure_osm_phase['Liq'])
    m.fs.pump.outlet.pressure[0].fix(osmotic_pressure * 1.5)
    m.fs.pump.initialize()
    propagate_state(m.fs.pump_to_RO)
    # print('The degrees of freedom are {}'.format(degrees_of_freedom(m)))
    # print('The osmotic pressure is {}'.format(osmotic_pressure/1e5))
    m.fs.RO.initialize()
    # Box solution
    m.fs.pump.outlet.pressure[0].unfix()
    m.fs.RO.recovery_vol_phase[0, 'Liq'].fix(0.4)
    print('DOF on RO before adding spacer porosity:', degrees_of_freedom(m.fs.RO))
    assert degrees_of_freedom(m) == 0
    solution = get_solver().solve(m, tee=False)
    assert_optimal_termination(solution)
    # m.fs.RO.report()
    print('Box problem solution: The operating pressure required to obtain {0} % recovery is {1:1.2f} bar'.format(
        m.fs.RO.recovery_vol_phase[0, 'Liq'].value * 100, m.fs.pump.outlet.pressure[0].value / 1e5))


def add_spacer_porosity_correlation(m):
    # New constraint for spacer porosity (porosity = f(angle,
    # filament thickness, mesh size, spacer height))
    m.fs.RO.angle = Var(initialize=30)  # Angle is defined in degrees
    m.fs.RO.angle.fix()
    m.fs.RO.mesh_size = Var(initialize=1e-3)
    m.fs.RO.mesh_size.fix()
    m.fs.RO.filament_dia = Var(initialize=2e-4)
    m.fs.RO.filament_dia.fix()
    m.fs.RO.spacer_height = Var(initialize=1e-3)
    m.fs.RO.spacer_height.fix()
    set_scaling_factor(m.fs.RO.spacer_height, 1e3)
    set_scaling_factor(m.fs.RO.angle, 0.01)
    set_scaling_factor(m.fs.RO.mesh_size, 1e3)
    set_scaling_factor(m.fs.RO.filament_dia, 1e4)
    assert (degrees_of_freedom(m)) == 0
    m.fs.RO.feed_side.porosity_constraint = (Constraint
                                             (expr=m.fs.RO.feed_side.spacer_porosity
                                                   == 1 - ((np.pi * m.fs.RO.filament_dia ** 2) / (
                                                     2 * m.fs.RO.mesh_size * m.fs.RO.spacer_height
                                                     * (pyomo.core.sin(np.pi * m.fs.RO.angle / 180))))
                                              )
                                             )
    m.fs.RO.feed_side.spacer_porosity.unfix()
    print('DOF on RO with the spacer porosity constraint:', degrees_of_freedom(m.fs.RO))
    print('DOF of flow sheet:', degrees_of_freedom(m))

    return m


def verify_spacer_constraint(m, init_angle, final_angle, iters):
    # Verifying the output from porosity constraint
    spacer_porosity = []
    angle_values = np.linspace(init_angle, final_angle, iters)
    for theta in angle_values:
        m.fs.RO.angle = theta
        # print('Value of the spacer angle is {}'.format(m.fs.RO.angle.value))
        calculate_variable_from_constraint(m.fs.RO.feed_side.spacer_porosity, m.fs.RO.feed_side.porosity_constraint)
        # print('Value of spacer porosity is {}'.format(m.fs.RO.feed_side.spacer_porosity.value))
        # print('Value of hydraulic diameter is {}'.format(m.fs.RO.feed_side.dh.value))
        spacer_porosity.append(m.fs.RO.feed_side.spacer_porosity.value)

    # fig, ax = plot.subplots()
    # ax.plot(angle_values, spacer_porosity, label='Spacer porosity')
    # ax.plot(angle_values, np.sin(np.pi * angle_values / 180), label='Sine')
    # ax.legend()

    # Solve the flow sheet with new constraint to check the effect of angle on Operating pressure
    angle_values = np.linspace(init_angle, final_angle, iters)
    spacer_porosity_vstheta = []
    dh_values_angle = []
    operP_angle = []
    for theta in angle_values:
        m.fs.RO.angle = theta
        # calculate_variable_from_constraint(m.fs.RO.feed_side.spacer_porosity,
        # m.fs.RO.feed_side.porosity_constraint)
        # print('The spacer porosity at an angle of {0} degrees is {1:1.2f}'.format(m.fs.RO.angle.value,
        # m.fs.RO.feed_side.spacer_porosity.value))
        assert (degrees_of_freedom(m)) == 0
        solver = get_solver()
        solution = solver.solve(m, tee=False)
        assert_optimal_termination(solution)
        print('Solved for spacer angle {0:2f}'.format(m.fs.RO.angle.value))
        spacer_porosity_vstheta.append(value(m.fs.RO.feed_side.spacer_porosity))
        dh_values_angle.append(value(m.fs.RO.feed_side.dh * 1e3))
        operP_angle.append(value(m.fs.RO.inlet.pressure[0] / 1e5))

    # Plot the values from solution
    fig, [ax1, ax2, ax3] = plot.subplots(1, 3, sharex=False, sharey=False)
    ax1.plot(angle_values, spacer_porosity_vstheta, color='red')
    ax1.set(xlabel='Spacer Angle (deg)', ylabel='Spacer Porosity')
    ax2.plot(angle_values, operP_angle, label='Operating pressure vs Spacer angle', color='blue')
    ax2.set(xlabel='Spacer Angle (deg)', ylabel='Operating Pressure (bar)')
    ax3.plot(angle_values, dh_values_angle, label='Hydraulic diameter vs Spacer angle', color='green')
    ax3.set(xlabel='Spacer Angle (deg)', ylabel='Hydraulic diameter (mm)')


def add_mass_transfer_correlation_DaCosta(m):
    m.fs.RO.feed_side.corr_factor_Dacosta = Var(initialize=1)
    m.fs.RO.feed_side.corr_factor_constraint = (Constraint
                                                (expr=m.fs.RO.feed_side.corr_factor_Dacosta
                                                      == 1.654 *
                                                      (m.fs.RO.filament_dia / m.fs.RO.feed_side.channel_height) ** 2
                                                      * m.fs.RO.feed_side.spacer_porosity ** 0.75
                                                      * pyomo.core.sin(np.pi * m.fs.RO.angle / 360) ** 0.086
                                                 ))

    calculate_variable_from_constraint(m.fs.RO.feed_side.corr_factor_Dacosta, m.fs.RO.feed_side.corr_factor_constraint)

    @m.fs.RO.feed_side.Constraint(
        [0],
        m.fs.RO.length_domain,
        m.fs.properties.solute_set,
        doc="Sherwood number DaCosta",
    )
    def eq_N_Sh_comp_DaCosta(b, t, x, j):
        return (
                b.N_Sh_comp[t, x, j]
                == (0.664 * m.fs.RO.feed_side.corr_factor_Dacosta * (b.N_Re[t, x] ** 0.5) *
                    (b.N_Sc_comp[t, x, j] ** 0.33) *
                    ((2 * m.fs.RO.feed_side.dh / m.fs.RO.mesh_size) ** 0.5))
        )

    print('DOF on RO with the DaCosta mass transfer constraint:', degrees_of_freedom(m.fs.RO))
    print('DOF of flow sheet:', degrees_of_freedom(m))

    return m


def verfiy_masstransfer_correlation(m):
    m.fs.RO.feed_side.eq_N_Sh_comp.deactivate()
    m.fs.RO.feed_side.eq_N_Sh_comp_DaCosta.activate()

    Guillen_sh = []
    DaCosta_sh = []
    module_position = []

    # grab Sherwood number values estimated using Guillen correlation
    for (t, x, j) in m.fs.RO.feed_side.N_Sh_comp:
        module_position.append(x)
        Guillen_sh.append(m.fs.RO.feed_side.N_Sh_comp[t, x, j].value)
    # grab Sherwood number values using DaCosta correlation
    for (t, x, j) in m.fs.RO.feed_side.N_Sh_comp:
        calculate_variable_from_constraint(m.fs.RO.feed_side.N_Sh_comp[t, x, j],
                                           m.fs.RO.feed_side.eq_N_Sh_comp_DaCosta[t, x, j])
        DaCosta_sh.append(m.fs.RO.feed_side.N_Sh_comp[t, x, j].value)

    import matplotlib.pyplot as plot
    fig, ax = plot.subplots()
    ax.plot(module_position, Guillen_sh, label='Guillen correlation')
    ax.plot(module_position, DaCosta_sh, label='DaCosta correlation')
    ax.set(xlabel='Module position (-)', ylabel='Sherwood number (-)')
    ax.legend()


def solve_with_DaCosta_mass_transfer_correlation(m, init_angle, final_angle, iters):
    m.fs.RO.feed_side.eq_N_Sh_comp.deactivate()
    m.fs.RO.feed_side.eq_N_Sh_comp_DaCosta.activate()
    print('DOF of the system with DaCosta constraint are {}'.format(degrees_of_freedom(m)))
    solver = get_solver()
    # log_infeasible_constraints(m, log_expression=True, log_variables=True)
    solution = solver.solve(m, tee=False)
    # Activate diagnostic toolbox
    # dt = DiagnosticsToolbox(m)
    # dt.report_structural_issues()
    # dt.display_components_with_inconsistent_units()
    # assert_units_consistent(m.fs.RO.feed_side.corr_factor_Dacosta)
    assert_optimal_termination(solution)

    # Solve the flow sheet with DaCosta mass transfer correlation
    # constraint to check the effect of angle on Operating pressure
    angle_values = np.linspace(init_angle, final_angle, iters)
    mass_transfer_coeff_values = []
    operP_angle_DaCosta = []
    length_values = []
    width_values = []
    for theta in angle_values:
        m.fs.RO.angle = theta
        assert (degrees_of_freedom(m)) == 0
        solver = get_solver()
        solution = solver.solve(m, tee=False)
        assert_optimal_termination(solution)
        print('Solved for spacer angle {0:2f}'.format(m.fs.RO.angle.value))
        operP_angle_DaCosta.append(value(m.fs.RO.inlet.pressure[0] / 1e5))
        mass_transfer_coeff_avg = np.average(
            [m.fs.RO.feed_side.N_Sh_comp[t, x, j].value for (t, x, j) in m.fs.RO.feed_side.N_Sh_comp])
        mass_transfer_coeff_values.append(value(mass_transfer_coeff_avg))
        length_values.append(m.fs.RO.length.value)
        width_values.append(m.fs.RO.width.value)

    # Plot the values from solution
    fig, [cx1, cx2, cx3, cx4] = plot.subplots(1, 4, sharex=False, sharey=False)
    cx1.plot(angle_values, operP_angle_DaCosta, label='Operating pressure vs Spacer angle', color='blue')
    cx1.set(xlabel='Spacer Angle (deg)', ylabel='Operating Pressure (bar)')
    cx2.plot(angle_values, mass_transfer_coeff_values, label='Mass Transfer Coeff. vs Spacer angle', color='red')
    cx2.set(xlabel='Spacer Angle (deg)', ylabel='Mass Transfer Coefficient (-)')
    cx3.plot(angle_values, length_values, label='Optimal length vs Spacer angle', color='purple')
    cx3.set(xlabel='Spacer Angle (deg)', ylabel='length (m)')
    cx4.plot(angle_values, width_values, label='Optimal width vs Spacer angle', color='green')
    cx4.set(xlabel='Spacer Angle (deg)', ylabel='width (m)')


def add_pressure_constraint_DaCosta(m):
    m.fs.RO.feed_side.NF_value = Var(initialize=1)

    # Normalized the NF value to unit length as the dp constraint in the model was for pressure drop across unit
    # length (Check if this assumption is correct)
    m.fs.RO.feed_side.NF_constraint = (Constraint
                                       (expr=m.fs.RO.feed_side.NF_value ==
                                             1 / (m.fs.RO.mesh_size + m.fs.RO.filament_dia *
                                                  pyomo.core.cos(np.pi * (m.fs.RO.angle / 360)))
                                        )
                                       )

    @m.fs.RO.feed_side.Constraint(
        [0],
        m.fs.RO.length_domain,
        doc="pressure change per unit length due to viscous drag + form drag + "
            "kinetic losses + viscous drag from walls based on DaCosta",
    )
    # Assuming the Cd value is 0.5 - need to develop a constraint for calculating it
    # Assuming the k_theta values in the third term to be 0. 5 for now but these should be replaced based on the
    # turbulent regime values given by Beek and Muttzal as described in page 100 of Da Costa's thesis
    def eq_dP_dx_DaCosta(b, t, x):
        return (
                - b.dP_dx[t, x]
                == 2.107 * (m.fs.RO.feed_side.NF_value / m.fs.RO.feed_side.area) *
                (m.fs.RO.filament_dia ** 3 * m.fs.RO.feed_side.velocity[t, x] ** 3 *
                 pyomo.core.sin(np.pi * m.fs.RO.angle / 360) * m.fs.RO.feed_side.properties[t, x].dens_mass_phase["Liq"]
                 * m.fs.RO.feed_side.properties[t, x].visc_d_phase["Liq"]) ** 0.5 +
                (0.5 * 0.5 * m.fs.RO.feed_side.NF_value * m.fs.RO.mesh_size * m.fs.RO.filament_dia *
                 m.fs.RO.feed_side.properties[t, x].dens_mass_phase['Liq'] * m.fs.RO.feed_side.velocity[t, x] ** 2 *
                 pyomo.core.sin(np.pi * m.fs.RO.angle / 360) ** 2) / m.fs.RO.feed_side.area +
                0.5 * m.fs.RO.feed_side.properties[t, x].dens_mass_phase['Liq'] * 0.5 * (m.fs.RO.feed_side.NF_value - 1)
                * ((m.fs.RO.mesh_size * m.fs.RO.spacer_height) / (2 * m.fs.RO.feed_side.area)) *
                (m.fs.RO.feed_side.velocity[t, x] / pyomo.core.cos(np.pi * m.fs.RO.angle / 360)) ** 0.5 +
                (12 * m.fs.RO.feed_side.velocity[t, x] * m.fs.RO.feed_side.properties[t, x].visc_d_phase['Liq'] *
                 m.fs.RO.feed_side.spacer_porosity) / (m.fs.RO.feed_side.channel_height ** 2)
        )

    print('DOF on RO with the DaCosta pressure constraint:', degrees_of_freedom(m.fs.RO))
    print('DOF of flowsheet:', degrees_of_freedom(m))

    return m


def check_pressure_constraint(m):
    dP_dx = []
    dP_dx_DaCosta = []
    module_position = []

    # grab dP_dx from the Darcy's equation
    for (t, x) in m.fs.RO.feed_side.eq_dP_dx:
        module_position.append(x)
        dP_dx.append(m.fs.RO.feed_side.dP_dx[t, x].value)
    # grab dP_dx from the DaCosta pressure equation
    for (t, x) in m.fs.RO.feed_side.eq_dP_dx_DaCosta:
        calculate_variable_from_constraint(m.fs.RO.feed_side.dP_dx[t, x],
                                           m.fs.RO.feed_side.eq_dP_dx_DaCosta[t, x])
        dP_dx_DaCosta.append(m.fs.RO.feed_side.dP_dx[t, x].value)

    import matplotlib.pyplot as plot
    fig, ax = plot.subplots()
    ax.plot(module_position, dP_dx, label='dP_dx_Darcy')
    ax.plot(module_position, dP_dx_DaCosta, label='dP_dx_DaCosta')
    ax.set(xlabel='Module position (-)', ylabel='dP-dx (Pa)')
    ax.legend()


def solve_with_DaCosta_dP_equation(m, init_angle, final_angle, iters):
    m.fs.RO.feed_side.eq_dP_dx.deactivate()
    m.fs.RO.feed_side.eq_dP_dx_DaCosta.activate()
    m.fs.RO.feed_side.eq_N_Sh_comp_DaCosta.activate()
    m.fs.RO.feed_side.eq_N_Sh_comp.deactivate()

    # try:
    for (t, x) in m.fs.RO.feed_side.eq_dP_dx_DaCosta:
        calculate_variable_from_constraint(m.fs.RO.feed_side.dP_dx[t, x],
                                           m.fs.RO.feed_side.eq_dP_dx_DaCosta[t, x])
    print('The DOF after activating the pressure correlation from DaCosta is:', degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 0
    solver = get_solver()
    solution = solver.solve(m, tee=False)
    assert_optimal_termination(solution)
    # except:
    # m.fs.display()
    # quit()

    angle_values = np.linspace(init_angle, final_angle, iters)
    operP_values = []
    for theta in angle_values:
        solution = solver.solve(m, tee=False)
        assert_optimal_termination(solution)
        m.fs.RO.angle.fix(theta)
        print('solved for the spacer angle of {}'.format(m.fs.RO.angle.value))
        operP_values.append(m.fs.RO.inlet.pressure[0].value)


if __name__ == '__main__':
    main()
