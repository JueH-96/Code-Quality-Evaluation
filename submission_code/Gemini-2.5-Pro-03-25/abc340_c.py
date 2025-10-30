# YOUR CODE HERE
import sys

# Setting a higher recursion depth limit is generally good practice 
# for competitive programming but not strictly necessary here because 
# the depth is O(log N), which is small (~60 for N=10^17).
# The default Python recursion depth limit (often 1000) is sufficient.
# sys.setrecursionlimit(2000) 

# Dictionary to store computed results for memoization
# This prevents re-calculating the cost for the same integer multiple times.
memo = {}

def C(N):
    """
    Calculates the total cost recursively starting from integer N, using memoization.
    
    The process involves repeatedly choosing an integer x >= 2, paying x yen,
    erasing x, and writing floor(x/2) and ceil(x/2). This continues until
    all integers are less than 2.

    The total cost C(N) follows the recurrence relation:
    C(N) = N + C(floor(N/2)) + C(ceil(N/2)) for N >= 2.
    The base cases are C(0) = 0 and C(1) = 0, as no operations are needed.
    
    Args:
        N: The integer currently being processed.

    Returns:
        The total cost associated with processing N and its descendants until all resulting numbers are < 2.
    """
    
    # Base case: If N is 0 or 1, no operations can be performed with this number. Cost is 0.
    if N <= 1: 
        return 0
    
    # Check memoization table for previously computed result for N.
    # If found, return it directly to avoid redundant computations.
    if N in memo:
        return memo[N]
    
    # Compute floor(N/2) and ceil(N/2) using integer arithmetic.
    # Python's // operator performs floor division.
    floor_N_half = N // 2
    
    # For positive integers N, ceil(N/2) can be computed as (N + 1) // 2.
    # This works correctly for both even and odd N.
    # Example: N=4 -> (4+1)//2 = 2. ceil(4/2)=2.
    # Example: N=5 -> (5+1)//2 = 3. ceil(5/2)=3.
    ceil_N_half = (N + 1) // 2 
    
    # Compute the result using the recursive formula.
    # The cost for the current operation step (processing N) is N yen.
    # To this, we add the costs from the recursive calls on the two resulting numbers:
    # floor(N/2) and ceil(N/2).
    result = N + C(floor_N_half) + C(ceil_N_half)
    
    # Store the computed result in the memoization table before returning.
    # Key is N, value is the computed cost C(N).
    memo[N] = result
    return result

# Read the single integer N from standard input.
# Using sys.stdin.readline() is efficient for potentially large inputs.
N = int(sys.stdin.readline())

# Compute the total cost by calling the recursive function C with the initial input N.
total_cost = C(N)

# Print the final computed total cost to standard output.
print(total_cost)