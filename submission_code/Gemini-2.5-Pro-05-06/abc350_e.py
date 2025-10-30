import sys

# Max recursion depth is ~60 for N=10^18, Python's default (1000-3000) is sufficient.
# sys.setrecursionlimit(2000) # This line is not strictly necessary.

def main():
    line = sys.stdin.readline().split()
    N_val = int(line[0])
    A_val = int(line[1])
    X_val = int(line[2])
    Y_val = int(line[3])

    memo = {} # Memoization dictionary

    # Nested function captures N_val, A_val, X_val, Y_val, and memo from outer scope.
    def solve(current_n):
        if current_n == 0:
            return 0.0
        
        # Check if result for current_n is already memoized
        if current_n in memo:
            return memo[current_n]

        # Option 1: Pay X_val, N becomes N // A_val
        # Cost is X_val plus expected cost from the new state.
        cost_opt1 = float(X_val) + solve(current_n // A_val)

        # Option 2: Pay Y_val, roll a die.
        # The expected cost if this path is chosen is derived as:
        # E_op2_val = (6/5) * Y_val + (1/5) * sum(solve(current_n // b) for b in [2,3,4,5,6])
        
        sum_expected_future_costs_op2_terms = 0.0
        # Sum E(current_n // b) for b = 2, 3, 4, 5, 6
        for b_die_outcome in range(2, 7): 
            sum_expected_future_costs_op2_terms += solve(current_n // b_die_outcome)
        
        # Calculate cost_opt2 using the formula
        cost_opt2 = (float(Y_val) * 6.0 + sum_expected_future_costs_op2_terms) / 5.0

        # Store the minimum of the two options in memo and return it
        memo[current_n] = min(cost_opt1, cost_opt2)
        return memo[current_n]

    result = solve(N_val)
    
    # Output precision requirement: absolute or relative error <= 10^-6.
    # Printing with a fixed number of decimal places, e.g., 10, is usually safe.
    sys.stdout.write(f"{result:.10f}
")

if __name__ == '__main__':
    main()