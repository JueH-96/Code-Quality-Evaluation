# YOUR CODE HERE
import math
import sys

# Helper function for ceiling division: computes ceil(a / b) for integers a, b.
# Assumes b > 0. Handles non-positive a correctly.
def ceil_div(a, b):
    # If a is non-positive, the number of sensors needed is 0.
    if a <= 0:
        return 0
    # Standard ceiling division for positive integers using integer arithmetic
    # equivalent to math.ceil(a / b)
    return (a + b - 1) // b

def solve():
    # Read input N
    N = int(sys.stdin.readline())
    # Read list of section lengths D
    D = list(map(int, sys.stdin.readline().split()))
    # Read parameters for sensor type 1
    L1_orig, C1_orig, K1_orig = map(int, sys.stdin.readline().split())
    # Read parameters for sensor type 2
    L2_orig, C2_orig, K2_orig = map(int, sys.stdin.readline().split())

    # Assign initial values which might be swapped later for optimization
    L1, C1, K1 = L1_orig, C1_orig, K1_orig
    L2, C2, K2 = L2_orig, C2_orig, K2_orig

    # Optimization: Ensure K1 (limit for the sensor type used as the DP state dimension) 
    # is the smaller limit to minimize DP state space size.
    # We designate the sensor type with the smaller K limit as "type-1" for the DP.
    swapped = False
    if K1 > K2:
        # Swap parameters of sensor types 1 and 2
        L1, C1, K1, L2, C2, K2 = L2, C2, K2, L1, C1, K1
        # Keep track that swap occurred. This is not strictly needed for the final calculation
        # using the logic below, but might be useful for debugging or alternative approaches.
        swapped = True 

    # Use a large integer value for infinity representation. 
    # Must be larger than any possible valid count or cost.
    # Max possible k2 count is K2 <= 1000. Max cost could be ~2 * 10^12. 
    # Using 10^18 is safe.
    INF = 10**18 

    # Initialize DP table. dp[k1_count] stores the minimum number of type-2 sensors 
    # needed across all processed sections, using exactly k1_count type-1 sensors in total.
    # The size of dp array is K1 + 1 (indices 0 to K1).
    dp = [INF] * (K1 + 1)
    # Base case: To cover 0 sections using 0 type-1 sensors, we need 0 type-2 sensors.
    dp[0] = 0 

    # Helper function to calculate minimum k2 needed for a single section of length D_val,
    # given k1i_count type-1 sensors are used for this section.
    # Uses L1, L2 which are the parameters for the current "type-1" and "type-2" sensors based on potential swap.
    def calculate_k2_needed(D_val, k1i_count, current_L1, current_L2):
      # Calculate remaining length after using k1i_count type-1 sensors
      rem_len = D_val - k1i_count * current_L1
      # Use ceiling division to find minimum number of type-2 sensors needed
      return ceil_div(rem_len, current_L2)

    # Main DP loop: iterate through each section
    for i in range(N):
        current_D = D[i] # Length of the current section
        
        # Precompute the required number of type-2 sensors (k2_needed) for the current section i
        # for all possible counts (k1i_count) of type-1 sensors used for this section, from 0 to K1.
        # req_k2_i[k1i_count] stores this minimum required k2 count.
        req_k2_i = [INF] * (K1 + 1)
        
        # Populate req_k2_i table
        for k1i_count in range(K1 + 1):
            # Calculate minimum k2 sensors needed for current section D with k1i_count type-1 sensors
            k2_needed = calculate_k2_needed(current_D, k1i_count, L1, L2) 

            # Check if the required number of type-2 sensors is within the limit K2
            if k2_needed <= K2:
                 # If valid, store the count. Otherwise, it remains INF.
                 req_k2_i[k1i_count] = k2_needed
                 
        # Prepare a new DP table for the state after processing section i
        # This avoids issues with using updated values within the same iteration.
        new_dp = [INF] * (K1 + 1)

        # Iterate through all possible previous total counts of type-1 sensors (prev_k1_total)
        # used for sections 0 to i-1. These states are stored in the current `dp` array.
        for prev_k1_total in range(K1 + 1): 
            # If the previous state dp[prev_k1_total] is INF, it means this state was unreachable. Skip.
            if dp[prev_k1_total] == INF:
                continue

            # Iterate through possible counts of type-1 sensors (k1i_for_current) used *specifically* for the current section i.
            # The loop range ensures that the new total count of type-1 sensors (current_total_k1) does not exceed K1.
            for k1i_for_current in range(K1 - prev_k1_total + 1): 
                
                # Calculate the new total count of type-1 sensors after processing section i
                current_total_k1 = prev_k1_total + k1i_for_current

                # Get the required k2 count for the current section i, given k1i_for_current type-1 sensors were used for it.
                k2_for_current = req_k2_i[k1i_for_current]
                
                # If this required k2 count is INF, it means using k1i_for_current type-1 sensors for section i is not possible
                # (either requires >K2 type-2 sensors or some other constraint violation). Skip this combination.
                if k2_for_current == INF: 
                   continue
                
                # Calculate the new total count of type-2 sensors after adding the sensors for section i.
                current_total_k2 = dp[prev_k1_total] + k2_for_current
                
                # Check if this new total count of type-2 sensors is within the limit K2.
                if current_total_k2 <= K2:
                    # If valid, update the `new_dp` state for `current_total_k1`.
                    # We take the minimum because there might be multiple ways to reach this state.
                    new_dp[current_total_k1] = min(new_dp[current_total_k1], current_total_k2)
        
        # After iterating through all possibilities, update the main dp table to reflect the state after processing section i.
        dp = new_dp

    # After processing all N sections, calculate the final minimum total cost.
    min_total_cost = INF

    # Iterate through all possible total counts of type-1 sensors (k1_total) from 0 to K1.
    for k1_total in range(K1 + 1):
        # Check if the state dp[k1_total] is reachable (not INF).
        if dp[k1_total] != INF: 
            # Get the minimum total count of type-2 sensors required for this k1_total count.
            k2_total = dp[k1_total]  
            
            # Calculate the total cost for this combination of k1_total type-1 sensors and k2_total type-2 sensors.
            # Use C1 and C2 which correspond to the costs of the current "type-1" and "type-2" sensor types
            # (these costs might have been swapped if K1 > K2 initially).
            cost = k1_total * C1 + k2_total * C2
            
            # Update the overall minimum total cost found so far.
            min_total_cost = min(min_total_cost, cost)

    # If min_total_cost remains INF, it means it's impossible to cover all sections within the constraints.
    if min_total_cost == INF:
        print("-1")
    else:
        # Otherwise, print the minimum total cost found.
        print(min_total_cost)

# Execute the main function to solve the problem
solve()