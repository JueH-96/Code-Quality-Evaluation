import sys
from scipy.optimize import linprog

def read_input():
    N, M = map(int, input().split())
    wheels = []
    for _ in range(N):
        wheel_data = list(map(int, input().split()))
        C, P = wheel_data[0], wheel_data[1]
        S = wheel_data[2:]
        wheels.append((C, P, S))
    return N, M, wheels

def solve(N, M, wheels):
    # Create the linear programming problem
    # Objective function coefficients (minimize cost)
    c = [wheel[0] for wheel in wheels] + [0]  # Add 0 for the dummy variable representing the expected value
    # Inequality constraints matrix
    A_ub = []
    # Inequality constraints vector
    b_ub = []
    # Equality constraints matrix (for the expected value)
    A_eq = [[0] * (N + 1) for _ in range(M + 1)]
    # Equality constraints vector
    b_eq = [0] * (M + 1)
    
    # Set up the constraints for each point total from 0 to M
    for points in range(M + 1):
        row = []
        for i, (C, P, S) in enumerate(wheels):
            # Calculate the expected value for this wheel
            expected_value = sum(max(0, points + s - M) for s in S) / P
            row.append(expected_value - 1)
        row.append(1)  # The dummy variable
        A_eq[points] = row
        b_eq[points] = 0
    
    # Solve the linear programming problem
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
    
    # Return the expected cost
    return res.fun

def main():
    N, M, wheels = read_input()
    expected_cost = solve(N, M, wheels)
    print(expected_cost)

if __name__ == "__main__":
    main()