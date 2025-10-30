import sys

# Memoization dictionary to store computed results for dp[n]
memo = {}

def solve(n: int, A: int, X: int, Y: int) -> float:
    """
    Calculates the minimum expected cost to reduce N to 0 using dynamic programming
    with memoization.

    Args:
        n: The current integer value to reduce to 0.
        A: The divisor for the first type of operation.
        X: The cost for the first type of operation.
        Y: The cost for the second type of operation (rolling a die).

    Returns:
        The minimum expected cost to reduce 'n' to 0.
    """
    if n == 0:
        return 0.0
    if n in memo:
        return memo[n]

    # --- Choice 1: Pay X yen, replace N with floor(N/A) ---
    # The cost for this operation is X.
    # The remaining expected cost is solve(n // A).
    cost_A_op = X + solve(n // A, A, X, Y)

    # --- Choice 2: Pay Y yen, roll a die ---
    # The die shows b from 1 to 6 with probability 1/6.
    # N is replaced with floor(N/b).
    #
    # Let E be the minimum expected cost for the current 'n'.
    # If we choose the die roll operation:
    # E = Y + (1/6) * E(floor(n/1)) + (1/6) * E(floor(n/2)) + ... + (1/6) * E(floor(n/6))
    # Since floor(n/1) is 'n', we have:
    # E = Y + (1/6) * E + (1/6) * sum(E(floor(n/b)) for b from 2 to 6)
    #
    # Solve for E:
    # (5/6) * E = Y + (1/6) * sum(E(floor(n/b)) for b from 2 to 6)
    # E = (6/5) * (Y + (1/6) * sum(E(floor(n/b)) for b from 2 to 6))
    # E = (6*Y + sum(E(floor(n/b)) for b from 2 to 6)) / 5

    sum_dp_b_other = 0.0
    # Sum up expected costs for die outcomes b = 2, 3, 4, 5, 6
    for b in range(2, 7): 
        sum_dp_b_other += solve(n // b, A, X, Y)
    
    # Calculate the total expected cost if we choose the die operation
    cost_die_op = (6.0 * Y + sum_dp_b_other) / 5.0

    # The minimum expected cost for 'n' is the minimum of the two choices
    result = min(cost_A_op, cost_die_op)
    
    # Store the result in memoization table
    memo[n] = result
    return result

def main():
    # Read input values
    N, A, X, Y = map(int, sys.stdin.readline().split())
    
    # Calculate the minimum expected cost starting from N
    ans = solve(N, A, X, Y)
    
    # Print the answer formatted to 15 decimal places for required precision
    print(f"{ans:.15f}")

if __name__ == "__main__":
    main()