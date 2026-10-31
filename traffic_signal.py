"""
Smart Traffic Signal Optimization - Dynamic Multi-Cycle Model
Incorporates:
  - Sets & Parameters structure (directions I, cycles t)
  - Vehicle update formula: V(i,t+1) = V(i,t) + A(i,t) - D(i,t)
  - Wait time update formula: W(i,t+1) = W(i,t) + T   (for non-green directions)
  - Repeating optimization loop (so it's a real dynamic system, not a one-shot pick)
"""

import random
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, PULP_CBC_CMD

# ---------- SETTINGS ----------
DIRECTIONS = ["North", "South", "East", "West"]
CYCLE_DURATION = 30      # T = seconds per cycle
NUM_CYCLES = 6           # how many rounds to simulate

def get_initial_state_from_user():
    """
    Ask the user for Cycle 1's numbers live, instead of hardcoding them.
    This proves the data isn't pre-written into the script.
    """
    print("Enter Cycle 1 traffic data (this is your live starting snapshot):\n")
    state = {}
    for d in DIRECTIONS:
        vehicles = int(input(f"{d} - number of vehicles waiting: "))
        wait = int(input(f"{d} - waiting time (seconds): "))
        state[d] = {"vehicles": vehicles, "wait": wait}
        print()
    return state


# ---------- INITIAL STATE (cycle 1 - typed in live, not hardcoded) ----------
state = get_initial_state_from_user()


def calculate_priority(state):
    """Priority = Vehicles x Wait time, for every direction."""
    for d in state:
        state[d]["priority"] = state[d]["vehicles"] * state[d]["wait"]
    return state


def solve_optimization(state):
    """PuLP picks exactly one direction to get the green signal."""
    problem = LpProblem("Traffic_Signal", LpMaximize)
    x = {d: LpVariable(f"x_{d}", cat="Binary") for d in DIRECTIONS}

    # Objective: maximize total priority of the chosen direction
    problem += lpSum(state[d]["priority"] * x[d] for d in DIRECTIONS)

    # Constraint: only one direction gets green
    problem += lpSum(x[d] for d in DIRECTIONS) == 1

    problem.solve(PULP_CBC_CMD(msg=0))  # msg=0 hides solver's verbose log

    winner = [d for d in DIRECTIONS if x[d].value() == 1][0]
    return winner


def update_state(state, winner):
    """
    Apply the two update formulas from the images:
      V(i,t+1) = V(i,t) + A(i,t) - D(i,t)
      W(i,t+1) = W(i,t) + T   (only for directions that did NOT get green)
    """
    for d in state:
        arriving = random.randint(2, 10)          # A(i,t): new cars showing up
        if d == winner:
            discharged = random.randint(30, 45)    # D(i,t) > 0: cars clear out
            state[d]["wait"] = max(5, state[d]["wait"] - CYCLE_DURATION)  # wait resets/drops
        else:
            discharged = 0                         # D(i,t) = 0: no cars leave
            state[d]["wait"] += CYCLE_DURATION      # wait keeps climbing

        new_vehicles = state[d]["vehicles"] + arriving - discharged
        state[d]["vehicles"] = max(0, new_vehicles)

    return state


# ---------- MAIN LOOP: runs multiple cycles automatically ----------
for cycle in range(1, NUM_CYCLES + 1):
    state = calculate_priority(state)
    winner = solve_optimization(state)

    print(f"\n--- Cycle {cycle} ---")
    for d in DIRECTIONS:
        print(f"{d:6s} | vehicles={state[d]['vehicles']:3d} "
              f"wait={state[d]['wait']:3d}s "
              f"priority={state[d]['priority']:5d}")
    print(f">>> Green signal goes to: {winner}")

    state = update_state(state, winner)
