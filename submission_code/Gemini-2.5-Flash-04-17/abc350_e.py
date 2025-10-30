import sys
import threading

# Increase recursion depth to handle the potentially deep recursive calls.
# The number of distinct reachable states (values of n) is the main factor for complexity.
# This number is expected to be manageable (polynomial in log N) for this type of problem.
# The maximum recursion depth is related to the number of steps to reduce N to 0,
# which is roughly log_min_factor(N). With factors >= 2, this is about log2(10^18) ~ 60.
# The call stack depth can be larger if the state space graph is deep, but it's likely
# bounded by a small multiple of log N. 10000 should be sufficient for typical scenarios.
# If Stack Overflow errors occur, this limit might need to be increased further,
# or a non-recursive approach (like explicit BFS state generation + iterative DP)
# or threading with increased stack size would be necessary.
sys.setrecursionlimit(10000)

memo = {}

def solve(n, A, X, Y):
    """
    Calculates the minimum expected cost to reduce n to 0.

    Args:
        n: The current integer value.
        A: The divisor for the fixed-cost operation.
        X: The cost of the fixed-cost operation.
        Y: The cost of the die-roll operation.

    Returns:
        The minimum expected cost.
    """
    if n == 0:
        return 0.0
    if n in memo:
        return memo[n]

    # Option 1: Replace N with floor(N/A) at cost X
    # Expected cost = X + E(floor(N/A))
    cost_opt1 = X + solve(n // A, A, X, Y)

    # Option 2: Roll a die at cost Y
    # The outcome b is 1..6 with probability 1/6. Replace N with floor(N/b).
    # Expected cost = Y + (1/6) * E(floor(N/1)) + (1/6) * E(floor(N/2)) + ... + (1/6) * E(floor(N/6))
    # E(floor(N/1)) = E(N).
    # Let S = sum_{b=2..6} E(floor(N/b)).
    # Expected cost = Y + (1/6) * E(N) + (1/6) * S.
    # If we choose option 2, the expected cost E(N) satisfies:
    # E(N) = Y + (1/6) * E(N) + (1/6) * S
    # (5/6) * E(N) = Y + (1/6) * S
    # E(N) = (6/5) * Y + (1/5) * S

    # Calculate the sum S = sum_{b=2..6} E(floor(N/b))
    sum_opt2_terms_b2_to_6 = 0.0
    for b in range(2, 7):
        sum_opt2_terms_b2_to_6 += solve(n // b, A, X, Y)

    # The expected cost if we choose option 2 is (6/5)*Y + (1/5)*S.
    # The minimum expected cost E(N) is the minimum of the cost if we choose option 1
    # and the expected cost if we choose option 2.
    # This is because the decision (which operation to choose) is made optimally
    # at each step *before* the outcome of the die roll is known.
    # The decision is between paying X to transition to state N//A, or paying Y to transition
    # to a state N//b where b is random, resulting in an expected cost of Y + E_avg_die.
    # E_avg_die = (1/6)Sum_{b=1..6} E(N//b) = (1/6)E(N) + (1/6)Sum_{b=2..6} E(N//b).
    # The optimal expected cost E(N) must satisfy:
    # E(N) = min(X + E(N//A), Y + (1/6)E(N) + (1/6)Sum_{b=2..6} E(N//b))
    # As derived in thought block, this leads to the formula:
    # E(N) = min(X + E(N//A), (6/5)Y + (1/5)Sum_{b=2..6} E(N//b)).

    cost_opt2_derived = (6.0/5.0) * Y + (1.0/5.0) * sum_opt2_terms_b2_to_6

    result = min(cost_opt1, cost_opt2_derived)
    memo[n] = result
    return result

def main():
    # Read input from stdin
    line = sys.stdin.readline().split()
    N = int(line[0])
    A = int(line[1])
    X = int(line[2])
    Y = int(line[3])

    # Call the recursive function to compute the minimum expected cost
    result = solve(N, A, X, Y)

    # Print the result with sufficient precision
    # 17 decimal places should be enough for the 10^-6 requirement.
    print(f"{result:.17f}")

# Standard boilerplate to run the main function
if __name__ == "__main__":
    # If dealing with very deep recursion on systems with small default stack size,
    # using threading might be necessary. However, increasing the global limit
    # is often sufficient in competitive programming environments.
    # Example using threading (uncomment if needed):
    # thread = threading.Thread(target=main)
    # thread.start()
    # thread.join()
    
    main()