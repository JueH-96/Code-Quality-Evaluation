# YOUR CODE HERE
import sys

# Set higher recursion depth limit for deep recursive calls
try:
    # Increase recursion depth limit. Needed for potentially deep recursion on large N.
    # N up to 10^18 suggests the number of recursive calls in a chain could be around log_2(N) ~ 60.
    # A higher limit like 2500 provides a safety margin against potential deeper chains or specific structures.
    sys.setrecursionlimit(2500) 
except Exception as e:
    # Silently ignore if setting limit fails (e.g., due to environment restrictions).
    # The program will try to run with the default limit. This might lead to RecursionError
    # on some systems/testcases if the default limit is too low.
    pass 

# Use a dictionary for memoization to store computed expected costs E(k).
# Key: integer state k, Value: float minimum expected cost E(k)
memo = {0: 0.0}

# Read input values N, A, X, Y from standard input
# N: initial integer value
# A: divisor for option 1
# X: cost for option 1
# Y: cost for option 2
N, A, X, Y = map(int, sys.stdin.readline().split())

# Define the recursive function to compute E(k) with memoization
def compute_E(k):
    """
    Computes the minimum expected cost E(k) to reach 0 starting from state k,
    using memoization to store and retrieve results for previously computed states.
    """
    # Base case: If k is 0, the cost is 0 because we have reached the target.
    if k == 0:
        return 0.0
    
    # Check if E(k) is already computed and stored in the memoization table.
    # If yes, return the stored value to avoid redundant computations. This is key for efficiency.
    if k in memo:
        return memo[k]

    # --- Calculate cost for Option 1 ---
    # Pay X yen, replace k with floor(k / A).
    # The total expected cost for this option is X plus the expected cost from the new state floor(k / A).
    # Recursively call compute_E for the resulting state k // A. (// is floor division)
    cost1 = X + compute_E(k // A)

    # --- Calculate cost for Option 2 ---
    # Pay Y yen, roll a die (outcome b in {1, ..., 6}), replace k with floor(k / b).
    # The initial expected cost equation is E(k) = Y + (1/6) * sum(E(floor(k/b)) for b=1..6).
    # This includes a term (1/6)*E(k) for b=1. We rearrange to solve for E(k):
    # (5/6) * E(k) = Y + (1/6) * sum(E(floor(k/b)) for b=2..6)
    # E(k) = (6/5) * Y + (1/5) * sum(E(floor(k/b)) for b=2..6)
    # Let's call this effective cost 'cost2'.
    
    # Calculate the sum of expected costs E(k // b) for b from 2 to 6.
    # This involves recursive calls for the states resulting from die rolls 2 through 6.
    cost2_sum_recursive = 0.0
    for b in range(2, 7): # Iterate b through {2, 3, 4, 5, 6}
        cost2_sum_recursive += compute_E(k // b)
    
    # Calculate the effective cost for Option 2 using the derived formula.
    # Ensure floating point arithmetic by using 6.0 and 5.0 to avoid potential integer division issues in some languages/versions
    # Python 3 handles division correctly, but explicit float promotes clarity.
    cost2 = (6.0 * Y + cost2_sum_recursive) / 5.0

    # The minimum expected cost E(k) is the minimum of the costs of the two options.
    # We choose the action that yields the lower expected cost.
    result = min(cost1, cost2)
    
    # Store the computed result in the memoization table before returning.
    # Key is the state k, value is the computed minimum expected cost E(k).
    memo[k] = result
    return result

# Start the computation by calling compute_E for the initial value N.
final_result = compute_E(N)

# Print the final result.
# Format the output to a high precision floating point number as required by the problem statement (at most 10^-6 error).
# Using .17f provides ample precision beyond the required minimum.
print(f"{final_result:.17f}")