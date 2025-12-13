import pandapower as pp
import pandapower.plotting as plot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection, PatchCollection
from pandapower.pypower.makeYbus import makeYbus
import pandas as pd
import numpy as np

# Alvaro, Domaboc, Orengo
print("=== PANDAPOWER NEWTON–RAPHSON POWER FLOW TEST ===")

# ------------------------------
#     Create empty network
# ------------------------------
net = pp.create_empty_network()

# ------------------------------
#     Define custom transformer types
# ------------------------------
pp.create_std_type(net, {
    "sn_mva": 100,
    "vn_hv_kv": 138,
    "vn_lv_kv": 69,
    "vk_percent": 12,
    "vkr_percent": 0.3997779628,
    "pfe_kw": 50,
    "i0_percent": 0.1,
    "shift_degree": 0
}, name="100 MVA 138/69 kV", element="trafo")

pp.create_std_type(net, {
    "sn_mva": 80,
    "vn_hv_kv": 138,
    "vn_lv_kv": 13.8,
    "vk_percent": 12,
    "vkr_percent": 0.3997779628,
    "pfe_kw": 50,
    "i0_percent": 0.1,
    "shift_degree": 0
}, name="80 MVA 138/13.8 kV", element="trafo")

pp.create_std_type(net, {
    "sn_mva": 25,
    "vn_hv_kv": 69,
    "vn_lv_kv": 13.8,
    "vk_percent": 9,
    "vkr_percent": 0.2998334721,
    "pfe_kw": 50,
    "i0_percent": 0.1,
    "shift_degree": 0
}, name="25 MVA 69/13.8 kV", element="trafo")

pp.create_std_type(net, {
    "sn_mva": 300,
    "vn_hv_kv": 230,
    "vn_lv_kv": 138,
    "vk_percent": 15,
    "vkr_percent": 0.3332510593,
    "pfe_kw": 50,
    "i0_percent": 0.1,
    "shift_degree": 0
}, name="300 MVA 230/138 kV", element="trafo")

# ------------------------------
#     Create buses
# ------------------------------
bus1 = pp.create_bus(net, vn_kv=138, name="UBAY")
bus2 = pp.create_bus(net, vn_kv=69, name="UBAY 2")
bus3 = pp.create_bus(net, vn_kv=13.8, name="UBAY 3")
bus4 = pp.create_bus(net, vn_kv=13.8, name="TAPAL")
bus5 = pp.create_bus(net, vn_kv=13.8, name="UBAY 4")
bus6 = pp.create_bus(net, vn_kv=69, name="CORELLA 3")
bus10 = pp.create_bus(net, vn_kv=138, name="CORELLA 2")
bus11 = pp.create_bus(net, vn_kv=230, name="CORELLA")
bus12 = pp.create_bus(net, vn_kv=230, name="MARIBOJOC CTS")

# ------------------------------
#     UBAY
# ------------------------------
pp.create_shunt(net, bus=bus1, q_mvar=-15, name="UBAY_S01")
pp.create_transformer(net,
                      hv_bus=bus1,
                      lv_bus=bus2,
                      std_type="100 MVA 138/69 kV",
                      name="UBAY_TR1")
pp.create_transformer(net,
                      hv_bus=bus1,
                      lv_bus=bus5,
                      std_type="80 MVA 138/13.8 kV",
                      name="BIDPP_TR1")

# ------------------------------
#     BUS 2
# ------------------------------
pp.create_gen(net, bus=bus2, p_mw=11.8, name="DAGSOL_G01")
pp.create_load(net, bus=bus2, p_mw=19.18, name="BOHOL_T1L1")
pp.create_load(net, bus=bus2, p_mw=19.23, name="UBAY_T1L1")
pp.create_load(net, bus=bus2, p_mw=3.89, name="UBAY_T2L2")
pp.create_load(net, bus=bus2, p_mw=0, name="BOHOL_T4L4")
pp.create_transformer(net,
                      hv_bus=bus2,
                      lv_bus=bus3,
                      std_type="25 MVA 69/13.8 kV",
                      name="UBBAT_TR1")
pp.create_transformer(net,
                      hv_bus=bus2,
                      lv_bus=bus4,
                      std_type="25 MVA 69/13.8 kV",
                      name="TAPAL_TR1")
pp.create_shunt(net, bus=bus2, q_mvar=5, name="UBAY_C01")
pp.create_shunt(net, bus=bus2, q_mvar=5, name="UBAY_C02")

# ------------------------------
#     BUS 3
# ------------------------------
pp.create_gen(net, bus=bus3, p_mw=0, name="UBAY_BAT")
pp.create_load(net, bus=bus3, p_mw=0, name="UBAY_BATL")

# ------------------------------
#     TAPAL
# ------------------------------
pp.create_gen(net, bus=bus4, p_mw=0.22, name="TPLPB4_U01")
pp.create_gen(net, bus=bus4, p_mw=0, name="TPLPB4_U02")
pp.create_gen(net, bus=bus4, p_mw=0, name="TPLPB4_U03")
pp.create_gen(net, bus=bus4, p_mw=0, name="TPLPB4_U04")
pp.create_load(net, bus=bus4, p_mw=0, name="UBAY_T3L3")

# ------------------------------
#     BUS 5
# ------------------------------
pp.create_gen(net, bus=bus5, p_mw=0, name="BIDPP_G01")

# ------------------------------
#     BUS 6
# ------------------------------
pp.create_load(net, bus=bus6, p_mw=7.7, name="COREL_T1L2")
pp.create_gen(net, bus=bus6, p_mw=0, name="JANOPO_G01")
pp.create_load(net, bus=bus6, p_mw=10.74, name="COREL_T1L1")
pp.create_shunt(net, bus=bus6, q_mvar=5, name="CRELLA_C01")
pp.create_shunt(net, bus=bus6, q_mvar=5, name="CRELLA_C02")
pp.create_shunt(net, bus=bus6, q_mvar=5, name="CRELLA_C03")
pp.create_shunt(net, bus=bus6, q_mvar=2.5, name="CRELLA_C04")
pp.create_shunt(net, bus=bus6, q_mvar=5, name="CRELLA_C05")
pp.create_shunt(net, bus=bus6, q_mvar=7.5, name="CRELLA_C06")
pp.create_shunt(net, bus=bus6, q_mvar=5, name="CRELLA_C07")
pp.create_shunt(net, bus=bus6, q_mvar=5, name="CRELLA_C08")
pp.create_gen(net, bus=bus6, p_mw=0, name="BDPP_U01")
pp.create_gen(net, bus=bus6, p_mw=0, name="BDPP_U02")
pp.create_gen(net, bus=bus6, p_mw=0, name="BDPP_U03")
pp.create_gen(net, bus=bus6, p_mw=0, name="BDPP_U04")
pp.create_gen(net, bus=bus6, p_mw=1.1, name="LOBOC_G01")
pp.create_gen(net, bus=bus6, p_mw=1.1, name="LOBOC_G03")
pp.create_gen(net, bus=bus6, p_mw=2.5, name="SEVILL_G01")
pp.create_load(net, bus=bus6, p_mw=64.12, name="COREL_T1L3")

# ------------------------------
#     BUS 10
# ------------------------------
pp.create_transformer(net,
                      hv_bus=bus10,
                      lv_bus=bus6,
                      std_type="100 MVA 138/69 kV",
                      name="CORELL_TR1")
pp.create_transformer(net,
                      hv_bus=bus10,
                      lv_bus=bus6,
                      std_type="100 MVA 138/69 kV",
                      name="CORELL_TR2")
pp.create_line_from_parameters(net, from_bus=bus1, to_bus=bus10, length_km=92.57,
                               r_ohm_per_km=0.0794, x_ohm_per_km=0.248,
                               c_nf_per_km=10.6945385, max_i_ka=1.2, name="UBAY_CRA1")

# ------------------------------
#     CORELLA
# ------------------------------
pp.create_transformer(net,
                      hv_bus=bus11,
                      lv_bus=bus10,
                      std_type="300 MVA 230/138 kV",
                      name="COREL_TR1")
pp.create_transformer(net,
                      hv_bus=bus11,
                      lv_bus=bus10,
                      std_type="300 MVA 230/138 kV",
                      name="COREL_TR2")
pp.create_load(net, bus=bus11, p_mw=0, name="CRA_SS")
pp.create_shunt(net, bus=bus11, p_mw=0.0, q_mvar=-70, name="CRA_S01")
pp.create_shunt(net, bus=bus11, p_mw=0.0, q_mvar=-70, name="CRA_S02")
pp.create_line_from_parameters(net, from_bus=bus11, to_bus=bus12, length_km=29,
                               r_ohm_per_km=0.0794, x_ohm_per_km=0.248,
                               c_nf_per_km=10.6945385, max_i_ka=1.2, name="CRLA_MJO1")
pp.create_line_from_parameters(net, from_bus=bus12, to_bus=bus12, length_km=29,
                               r_ohm_per_km=0.0794, x_ohm_per_km=0.248,
                               c_nf_per_km=10.6945385, max_i_ka=1.2, name="CRLA_MJO2")

# ------------------------------
#    External Grid
# ------------------------------
pp.create_ext_grid(net, bus=bus1, vm_pu=1.0, name="External Grid (UBAY)")
pp.create_ext_grid(net, bus=bus12, vm_pu=1.0, name="External Grid (MARIBOJOC)")

# ------------------------------
#     Add bus coordinates for SLD
# ------------------------------
coords = {
    bus1: (12, 5),        # UBAY
    bus2: (11, 3.5),      # Bus 2
    bus3: (9.5, -2),      # Bus 3
    bus4: (16, 1),        # TAPAL
    bus5: (12, -4),       # Bus 5
    bus6: (7, 1),         # Bus 6
    bus10: (5, 2),        # Bus 10
    bus11: (4, 4),        # CORELLA
    bus12: (0, 0)         # MARIBOJOC CTS
}

import pandas as pd

# Initialize geodata for all buses
net.bus_geodata = pd.DataFrame(columns=['x', 'y'], dtype=float)
for bus in net.bus.index:
    if bus in coords:
        x, y = coords[bus]
        net.bus_geodata.loc[bus] = [x, y]
    else:
        net.bus_geodata.loc[bus] = [0.0, 0.0]

net.bus_geodata = net.bus_geodata.fillna(0.0)

# ------------------------------
#    Run Power Flow (Newton-Raphson)
# ------------------------------
print("\nRunning Newton–Raphson Power Flow...\n")

import time
start_time = time.time()

pp.runpp(net, algorithm='nr', init='auto', calculate_voltage_angles=True, tolerance_mva=1e-6,
         max_iteration=10, numba=True, enforce_q_lims=True)

convergence_time = time.time() - start_time

print("\nNewton–Raphson Power Flow Converged!")
print(f"Convergence Time: {convergence_time:.6f} seconds")

# ------------------------------
#    Display Results
# ------------------------------
print("\n=== BUS VOLTAGE RESULTS (p.u.) ===")
print(net.res_bus[["vm_pu", "va_degree"]])

print("\n=== LINE FLOW RESULTS ===")
print(net.res_line[["loading_percent", "p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]])

# ------------------------------
#    Calculate and Display Ybus and Jacobian Matrix
# ------------------------------

def calculate_ybus_and_jacobian(net):
    """
    Calculate the Ybus and Jacobian matrix from the power flow results
    """
    from pandapower.pypower.makeYbus import makeYbus
    
    # Get the internal ppc (PYPOWER case format)
    ppc = net._ppc
    baseMVA = ppc["baseMVA"]
    bus = ppc["bus"]
    branch = ppc["branch"]
    
    # Get Ybus
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    Ybus = Ybus.toarray()
    
    # Get voltage results
    V = net.res_bus['vm_pu'].values
    theta = np.deg2rad(net.res_bus['va_degree'].values)
    
    # Number of buses
    n = len(V)
    
    # Separate PQ and PV buses (exclude slack buses)
    pq_buses = []
    pv_buses = []
    
    for idx in range(n):
        bus_type = int(bus[idx, 1])  # Bus type column
        if bus_type == 1:  # PQ bus
            pq_buses.append(idx)
        elif bus_type == 2:  # PV bus
            pv_buses.append(idx)
    
    # Number of PQ and PV buses
    npq = len(pq_buses)
    npv = len(pv_buses)
    
    # Jacobian matrix size
    n_equations = npv + 2 * npq
    J = np.zeros((n_equations, n_equations))
    
    # Create complex voltage vector
    V_complex = V * np.exp(1j * theta)
    
    # Calculate power injections for verification
    S = V_complex * np.conj(Ybus @ V_complex)
    
    # Build Jacobian matrix
    # J = [dP/dTheta  dP/dV  ]
    #     [dQ/dTheta  dQ/dV  ]
    
    all_buses = pv_buses + pq_buses
    
    for i, bus_i in enumerate(all_buses):
        for j, bus_j in enumerate(all_buses):
            # dP/dTheta (H matrix)
            if i == j:
                # Diagonal elements
                J[i, j] = -V[bus_i]**2 * Ybus[bus_i, bus_i].imag - V[bus_i] * sum(
                    V[k] * (Ybus[bus_i, k].real * np.sin(theta[bus_i] - theta[k]) -
                            Ybus[bus_i, k].imag * np.cos(theta[bus_i] - theta[k]))
                    for k in range(n) if k != bus_i
                )
            else:
                J[i, j] = V[bus_i] * V[bus_j] * (
                    Ybus[bus_i, bus_j].real * np.sin(theta[bus_i] - theta[bus_j]) -
                    Ybus[bus_i, bus_j].imag * np.cos(theta[bus_i] - theta[bus_j])
                )
    
    # dP/dV (N matrix) - only for PQ buses
    for i, bus_i in enumerate(all_buses):
        for j, bus_j in enumerate(pq_buses):
            col = npv + npq + j
            if bus_i == bus_j:
                # Diagonal
                J[i, col] = V[bus_i] * Ybus[bus_i, bus_i].real + sum(
                    V[k] * (Ybus[bus_i, k].real * np.cos(theta[bus_i] - theta[k]) +
                            Ybus[bus_i, k].imag * np.sin(theta[bus_i] - theta[k]))
                    for k in range(n) if k != bus_i
                )
            else:
                J[i, col] = V[bus_i] * (
                    Ybus[bus_i, bus_j].real * np.cos(theta[bus_i] - theta[bus_j]) +
                    Ybus[bus_i, bus_j].imag * np.sin(theta[bus_i] - theta[bus_j])
                )
    
    # dQ/dTheta (M matrix) - only for PQ buses
    for i, bus_i in enumerate(pq_buses):
        row = npv + npq + i
        for j, bus_j in enumerate(all_buses):
            if bus_i == bus_j:
                # Diagonal
                J[row, j] = -V[bus_i]**2 * Ybus[bus_i, bus_i].real + V[bus_i] * sum(
                    V[k] * (Ybus[bus_i, k].real * np.cos(theta[bus_i] - theta[k]) +
                            Ybus[bus_i, k].imag * np.sin(theta[bus_i] - theta[k]))
                    for k in range(n) if k != bus_i
                )
            else:
                J[row, j] = -V[bus_i] * V[bus_j] * (
                    Ybus[bus_i, bus_j].real * np.cos(theta[bus_i] - theta[bus_j]) +
                    Ybus[bus_i, bus_j].imag * np.sin(theta[bus_i] - theta[bus_j])
                )
    
    # dQ/dV (L matrix) - only for PQ buses
    for i, bus_i in enumerate(pq_buses):
        row = npv + npq + i
        for j, bus_j in enumerate(pq_buses):
            col = npv + npq + j
            if bus_i == bus_j:
                # Diagonal
                J[row, col] = -V[bus_i] * Ybus[bus_i, bus_i].imag + sum(
                    V[k] * (Ybus[bus_i, k].real * np.sin(theta[bus_i] - theta[k]) -
                            Ybus[bus_i, k].imag * np.cos(theta[bus_i] - theta[k]))
                    for k in range(n) if k != bus_i
                )
            else:
                J[row, col] = V[bus_i] * (
                    Ybus[bus_i, bus_j].real * np.sin(theta[bus_i] - theta[bus_j]) -
                    Ybus[bus_i, bus_j].imag * np.cos(theta[bus_i] - theta[bus_j])
                )
    
    return Ybus, J, all_buses, pq_buses, pv_buses

# Calculate Ybus and Jacobian
Ybus, J, all_buses, pq_buses, pv_buses = calculate_ybus_and_jacobian(net)

# Get bus numbers
bus_numbers = list(range(len(net.bus)))

# ------------------------------
#    Display Ybus Matrix
# ------------------------------
print("\n=== YBUS MATRIX (Complex Form) ===")
ybus_df = pd.DataFrame(
    Ybus,
    index=bus_numbers,
    columns=bus_numbers
)

# Set display options for better formatting
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.precision', 4)

print(ybus_df)

# ------------------------------
#    Display Jacobian Matrix
# ------------------------------

# Create labels for Jacobian matrix
row_labels = []
col_labels = []

# Column labels: θ for all non-slack, then V for PQ buses
for bus_idx in all_buses:
    col_labels.append(f"θ_{bus_idx}")
for bus_idx in pq_buses:
    col_labels.append(f"V_{bus_idx}")

# Row labels: P for all non-slack, then Q for PQ buses
for bus_idx in all_buses:
    row_labels.append(f"P_{bus_idx}")
for bus_idx in pq_buses:
    row_labels.append(f"Q_{bus_idx}")

print("\n=== JACOBIAN MATRIX (at convergence) ===")
jacobian_df = pd.DataFrame(
    J,
    index=row_labels,
    columns=col_labels
)

print(jacobian_df)

# Display additional information
# print(f"\nJacobian Matrix Size: {J.shape[0]} x {J.shape[1]}")
# print(f"Condition Number: {np.linalg.cond(J):.4e}")
# print(f"PV Buses: {len(pv_buses)}, PQ Buses: {len(pq_buses)}")

print("\n=== POWER FLOW METHOD: Newton–Raphson ===")
# ------------------------------
#    Setup OPF with generator costs and limits
# ------------------------------
print("\n=== SETTING UP OPF ===")

# To test functionality only
for idx in net.gen.index:
    current_p = net.gen.loc[idx, 'p_mw']
    # Set min to 0 and max to 150% of current or at least 10 MW
    net.gen.loc[idx, 'min_p_mw'] = max(0.5, 0.1 * current_p)  # 10% of nominal or at least 0.5 MW
    net.gen.loc[idx, 'max_p_mw'] = max(current_p * 1.5, 10.0)
    net.gen.loc[idx, 'controllable'] = True  # Make generators controllable by OPF

# Set reactive power limits for generators
for idx in net.gen.index:
    net.gen.loc[idx, 'min_q_mvar'] = -50.0
    net.gen.loc[idx, 'max_q_mvar'] = 50.0

# Add polynomial costs for all generators
# Format: create_poly_cost(net, element_index, element_type, cost_array)
# cost_array = [c2, c1, c0] for cost = c2*P^2 + c1*P + c0

for idx in net.gen.index:
    # Sample cost: 100 PHP/MWh (linear cost)
    # Using [0, 100, 0] means cost = 100*P
    pp.create_poly_cost(net, idx, 'gen', cp1_eur_per_mw=50 + idx*10, cp0_eur=0)

# Add costs for external grids (typically lower to represent grid supply)
for idx in net.ext_grid.index:
    pp.create_poly_cost(net, idx, 'ext_grid', cp1_eur_per_mw=50, cp0_eur=0)

print("Generator costs added successfully.")

# ------------------------------
#    Run OPF
# ------------------------------
print("\n=== RUNNING OPF (AC Optimal Power Flow) ===")

try:
    pp.runopp(net, ac=True, verbose=True, delta=1e-10)
    print("\nOPF converged successfully!")
    
    # Extract LMP data
    print("\n=== LOCATIONAL MARGINAL PRICES (LMP) ===")
    print("\nLMP for each bus (PHP/MWh):")
    print(net.res_bus[['vm_pu', 'va_degree', 'p_mw', 'q_mvar', 'lam_p', 'lam_q']])
    
    if 'lambda_p' in net.res_bus.columns:
        print("\n=== LMP Summary (using lambda_p) ===")
        lmp_summary = net.res_bus[['vm_pu', 'lambda_p']].copy()
        lmp_summary.columns = ['Voltage (pu)', 'LMP (PHP/MWh)']
        print(lmp_summary)
    
    # Create a clean LMP dataframe
    print("\n=== CLEAN LMP RESULTS ===")
    lmp_results = pd.DataFrame({
        'Bus': net.bus['name'],
        'Voltage_kV': net.bus['vn_kv'],
        'LMP_PHP_per_MWh': net.res_bus['lam_p'].values
    })
    print(lmp_results)
    
    # Show generator dispatch results
    print("\n=== GENERATOR DISPATCH (OPF) ===")
    print(net.res_gen[['p_mw', 'q_mvar', 'va_degree', 'vm_pu']])
    
    print("\n=== EXTERNAL GRID RESULTS ===")
    print(net.res_ext_grid[['p_mw', 'q_mvar']])
    
except Exception as e:
    print(f"\nOPF failed to converge or error occurred: {e}")
    print("Falling back to regular power flow results...")
    # The regular power flow results are already available from earlier

# ------------------------------
#    Create Detailed Single-Line Diagram
# ------------------------------

def plot_detailed_sld(net, convergence_time):
    """Create a detailed single-line diagram with horizontal element layout"""

    fig, ax = plt.subplots(figsize=(22, 14))

    bus_coords = net.bus_geodata

    # Draw transmission lines
    for idx, line in net.line.iterrows():
        from_bus = line['from_bus']
        to_bus = line['to_bus']
        x1, y1 = bus_coords.loc[from_bus, 'x'], bus_coords.loc[from_bus, 'y']
        x2, y2 = bus_coords.loc[to_bus, 'x'], bus_coords.loc[to_bus, 'y']
        ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, zorder=1)

    # Draw transformers
    for idx, trafo in net.trafo.iterrows():
        hv_bus = trafo['hv_bus']
        lv_bus = trafo['lv_bus']
        x1, y1 = bus_coords.loc[hv_bus, 'x'], bus_coords.loc[hv_bus, 'y']
        x2, y2 = bus_coords.loc[lv_bus, 'x'], bus_coords.loc[lv_bus, 'y']

        ax.plot([x1, x2], [y1, y2], 'g-', linewidth=2, zorder=2)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        circle1 = mpatches.Circle((mid_x - 0.05, mid_y), 0.08, color='green', fill=False, linewidth=1, zorder=3)
        circle2 = mpatches.Circle((mid_x + 0.05, mid_y), 0.08, color='green', fill=False, linewidth=1, zorder=3)
        ax.add_patch(circle1)
        ax.add_patch(circle2)

    # Draw buses
    for bus in net.bus.index:
        x, y = bus_coords.loc[bus, 'x'], bus_coords.loc[bus, 'y']
        circle = mpatches.Circle((x, y), 0.08, color='black', zorder=10)
        ax.add_patch(circle)

    # Count elements per bus to calculate total width needed
    bus_elements = {}
    for bus in net.bus.index:
        loads = net.load[net.load.bus == bus]
        gens = net.gen[net.gen.bus == bus]
        shunts = net.shunt[net.shunt.bus == bus]
        bus_elements[bus] = {
            'loads': loads,
            'gens': gens,
            'shunts': shunts,
            'total': len(loads) + len(gens) + len(shunts)
        }

    # Spacing parameters
    element_spacing = 0.6   # Horizontal spacing between elements (increased for text)
    vertical_drop = 0.7     # How far below bus to place elements

    # Draw elements for each bus
    for bus in net.bus.index:
        x_bus, y_bus = bus_coords.loc[bus, 'x'], bus_coords.loc[bus, 'y']
        elements = bus_elements[bus]
        total_elements = elements['total']

        if total_elements == 0:
            continue

        # Calculate starting x position to center elements under bus
        total_width = (total_elements - 1) * element_spacing
        x_start = x_bus - total_width / 2
        current_position = 0

        # Draw loads
        for idx, load in elements['loads'].iterrows():
            x_elem = x_start + current_position * element_spacing
            y_elem = y_bus - vertical_drop

            # Draw triangle (load symbol)
            triangle = mpatches.Polygon([
                [x_elem - 0.08, y_elem + 0.08],
                [x_elem + 0.08, y_elem + 0.08],
                [x_elem, y_elem - 0.08]
            ], closed=True, color='red', zorder=5)
            ax.add_patch(triangle)

            # Draw connection line
            ax.plot([x_bus, x_elem], [y_bus, y_elem], 'k-', linewidth=1, zorder=4)

            # Add label with smaller font and better positioning
            label = f"{load['name']}\n{load.p_mw:.1f}MW"
            ax.text(x_elem, y_elem - 0.25, label, fontsize=5.5, ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

            current_position += 1

        # Draw generators
        for idx, gen in elements['gens'].iterrows():
            x_elem = x_start + current_position * element_spacing
            y_elem = y_bus - vertical_drop

            # Draw circle (generator symbol)
            circle = mpatches.Circle((x_elem, y_elem), 0.1, facecolor='green', 
                                    edgecolor='darkgreen', linewidth=1.5, zorder=5)
            ax.add_patch(circle)

            # Draw connection line
            ax.plot([x_bus, x_elem], [y_bus, y_elem], 'k-', linewidth=1, zorder=4)

            # Add label with smaller font and better positioning
            label = f"{gen['name']}\n{gen.p_mw:.1f}MW"
            ax.text(x_elem, y_elem - 0.25, label, fontsize=5.5, ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

            current_position += 1

        # Draw shunts/capacitors
        for idx, shunt in elements['shunts'].iterrows():
            x_elem = x_start + current_position * element_spacing
            y_elem = y_bus - vertical_drop

            # Draw capacitor symbol (two parallel lines)
            ax.plot([x_elem - 0.05, x_elem - 0.05], [y_elem - 0.08, y_elem + 0.08],
                   'b-', linewidth=2, zorder=5)
            ax.plot([x_elem + 0.05, x_elem + 0.05], [y_elem - 0.08, y_elem + 0.08],
                   'b-', linewidth=2, zorder=5)

            # Draw connection line
            ax.plot([x_bus, x_elem], [y_bus, y_elem], 'k-', linewidth=1, zorder=4)

            # Add label with smaller font and better positioning
            label = f"{shunt['name']}\n{shunt.q_mvar:.1f}MVar"
            ax.text(x_elem, y_elem - 0.25, label, fontsize=5.5, ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

            current_position += 1

    # Draw external grids
    for idx, ext_grid in net.ext_grid.iterrows():
        bus = ext_grid.bus
        x = bus_coords.loc[bus, 'x']
        y = bus_coords.loc[bus, 'y']

        # Draw external grid symbol (three horizontal lines)
        y_offset = 0.5
        for i in range(3):
            ax.plot([x - 0.15, x + 0.15], [y + y_offset + 0.05 * i, y + y_offset + 0.05 * i],
                   'k-', linewidth=2, zorder=5)
        ax.plot([x, x], [y, y + y_offset], 'k-', linewidth=2, zorder=4)

        # Add label
        ax.text(x, y + y_offset + 0.2, ext_grid['name'], fontsize=8, ha='center', va='bottom',
               fontweight='bold')

    # Add bus labels
    for bus in net.bus.index:
        x = bus_coords.loc[bus, 'x']
        y = bus_coords.loc[bus, 'y']
        name = net.bus.loc[bus, 'name']
        voltage = net.res_bus.loc[bus, 'vm_pu']

        ax.text(x, y + 0.25, f"{name}\n{voltage:.3f}pu", fontsize=9, ha='center', va='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Create legend with actual symbols
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
               markersize=8, label='Bus'),
        Line2D([0], [0], color='blue', linewidth=2, label='Line'),
        Line2D([0], [0], color='green', linewidth=2, label='Transformer'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', 
               markersize=8, markeredgewidth=0, label='Load'),
        Line2D([0], [0], marker='o', color='darkgreen', markerfacecolor='green', 
               markersize=8, markeredgewidth=1.5, label='Generator'),
        Line2D([0], [0], marker='$||$', color='blue', markersize=12, 
               markeredgewidth=2.5, label='Shunt'),
        Line2D([0], [0], marker='_', color='black', markersize=10, 
               markeredgewidth=2, label='Ext Grid')
    ]
    
    ax.legend(handles=legend_elements, loc='lower center', fontsize=8, 
             framealpha=0.9, ncol=7, columnspacing=1.0, 
             handlelength=1.5, handleheight=1.0, borderpad=0.5)

    # Add power flow results box in lower left
    # Bus voltage results
    bus_voltage_text = "=== BUS VOLTAGE RESULTS (p.u.) ===\n"
    bus_voltage_text += f"{'Bus':<4} {'vm_pu':<10} {'va_degree':<12}\n"
    for bus in net.bus.index:
        vm_pu = net.res_bus.loc[bus, 'vm_pu']
        va_deg = net.res_bus.loc[bus, 'va_degree']
        bus_voltage_text += f"{bus:<4} {vm_pu:<10.6f} {va_deg:<12.6f}\n"
    
    # Line flow results
    line_flow_text = "\n=== LINE FLOW RESULTS ===\n"
    line_flow_text += f"{'Line':<4} {'loading_%':<12} {'p_from_mw':<12} {'q_from_mvar':<14} {'p_to_mw':<12} {'q_to_mvar':<12}\n"
    for idx, line in net.line.iterrows():
        loading = net.res_line.loc[idx, 'loading_percent']
        p_from = net.res_line.loc[idx, 'p_from_mw']
        q_from = net.res_line.loc[idx, 'q_from_mvar']
        p_to = net.res_line.loc[idx, 'p_to_mw']
        q_to = net.res_line.loc[idx, 'q_to_mvar']
        line_flow_text += f"{idx:<4} {loading:<12.6f} {p_from:<12.6f} {q_from:<14.6f} {p_to:<12.6f} {q_to:<12.6f}\n"
    
    results_text = f"""Newton-Raphson Power Flow Converged!
Convergence Time: {convergence_time:.6f} seconds

{bus_voltage_text}{line_flow_text}"""
    
    ax.text(0.01, 0.08, results_text, transform=ax.transAxes,
           fontsize=6, verticalalignment='bottom', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                    edgecolor='black', linewidth=1),
           family='monospace')

    ax.set_aspect('equal')
    ax.set_title('Bohol Subgrid Model', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Coordinate', fontsize=10)
    ax.set_ylabel('Y Coordinate', fontsize=10)

    plt.tight_layout()
    plt.show()

# Create the detailed plot
plot_detailed_sld(net, convergence_time)
