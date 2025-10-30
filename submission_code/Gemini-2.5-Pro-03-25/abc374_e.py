# YOUR CODE HERE
import sys

# Function for ceiling division ceil(a / b)
# Handles a <= 0 correctly, returns 0.
# Assumes b > 0, which is guaranteed by constraints A_i, B_i >= 1.
def ceil_div(a, b):
    """Computes ceil(a / b) for integers a and b, where b > 0.
    Returns 0 if a <= 0."""
    # If the required capacity 'a' is non-positive, 0 machines are needed.
    if a <= 0:
        return 0
    # Calculate ceil(a / b) using integer arithmetic.
    # The formula (a + b - 1) // b correctly computes ceiling division for positive b.
    return (a + b - 1) // b

# Main solving function
def solve():
    """Reads input, solves the problem using binary search on the answer, and prints the result."""
    N, X = map(int, sys.stdin.readline().split())
    
    processes = []
    for _ in range(N):
        # Read process parameters: Ai, Pi, Bi, Qi
        processes.append(list(map(int, sys.stdin.readline().split())))

    # Check function: determines if a production capacity K is achievable within budget X
    def check(K):
        """Checks if capacity K is achievable within budget X by finding the minimum cost."""
        # Base case: capacity 0 is always achievable with cost 0.
        if K == 0: 
            return True
            
        total_min_cost = 0
        
        # Calculate the minimum cost for each process to achieve capacity K
        for i in range(N):
            Ai, Pi, Bi, Qi = processes[i]
            
            # Initialize minimum cost for this process to infinity.
            # float('inf') works correctly with integer comparisons and min().
            current_min_cost_i = float('inf') 

            # The core optimization relies on a property derived from the problem structure:
            # An optimal (minimum cost) configuration (s_i, t_i) to achieve capacity >= K exists such that:
            # - if Bi * Pi <= Ai * Qi, then t_i < Ai.
            # - if Bi * Pi > Ai * Qi, then s_i < Bi.
            # This bounds one of the variables (s_i or t_i) by at most max(Ai, Bi) <= 100.
            
            # Check the condition Bi * Pi <= Ai * Qi to decide the iteration strategy
            if Bi * Pi <= Ai * Qi:
                 # Case 1: Iterate through possible values of t_i from 0 to A_i - 1.
                 # Since A_i >= 1, limit_t is at least 1. The range is at most 100 elements.
                 limit_t = Ai 
                 for ti in range(limit_t):
                     # Calculate the remaining capacity needed from S machines after using ti T machines.
                     needed_K_from_S = K - ti * Bi
                     # Calculate the minimum number of S machines required using ceiling division.
                     si = ceil_div(needed_K_from_S, Ai)
                     # Calculate the total cost for this (si, ti) configuration.
                     cost = si * Pi + ti * Qi
                     # Update the minimum cost found so far for process i.
                     current_min_cost_i = min(current_min_cost_i, cost)
                 
            else: # Bi * Pi > Ai * Qi
                 # Case 2: Iterate through possible values of s_i from 0 to B_i - 1.
                 # Since B_i >= 1, limit_s is at least 1. The range is at most 100 elements.
                 limit_s = Bi
                 for si in range(limit_s):
                     # Calculate the remaining capacity needed from T machines after using si S machines.
                     needed_K_from_T = K - si * Ai
                     # Calculate the minimum number of T machines required using ceiling division.
                     ti = ceil_div(needed_K_from_T, Bi)
                     # Calculate the total cost for this (si, ti) configuration.
                     cost = si * Pi + ti * Qi
                     # Update the minimum cost found so far for process i.
                     current_min_cost_i = min(current_min_cost_i, cost)

            # Add the minimum cost for process i to the total cumulative cost.
            # Python's arbitrary precision integers handle potentially large costs.
            total_min_cost += current_min_cost_i
            
            # Optimization: If at any point the cumulative cost exceeds the budget,
            # then capacity K is not achievable. Stop early.
            if total_min_cost > X:
                return False
        
        # If the loop completes without exceeding the budget, capacity K is achievable.
        return total_min_cost <= X

    # Binary search for the maximum achievable production capacity K
    low = 0
    # Set a sufficiently large upper bound for K. 
    # Max capacity could be around 10^9 based on X=10^7, A=100, P=1.
    # Using 10^10 provides a safe margin.
    high = 10**10 
    
    ans = 0 # Stores the maximum K found that is achievable
    
    # Standard binary search implementation
    while low <= high:
        mid = (low + high) // 2
        
        if check(mid):
            # If capacity `mid` is achievable, it becomes a potential answer.
            # We then try to find if an even higher capacity is possible.
            ans = mid 
            low = mid + 1
        else:
            # If capacity `mid` is not achievable, the target capacity must be lower.
            high = mid - 1
            
    # Print the final answer, which is the largest K found to be achievable.
    print(ans)

# Execute the main function to solve the problem
solve()