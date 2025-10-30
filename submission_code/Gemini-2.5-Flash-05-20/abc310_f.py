import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    # dp[mask] stores the number of ways to assign values to dice processed so far
    # such that no non-empty subset sum to 10, and 'mask' represents the set of
    # achievable non-empty subset sums from 1 to 9.
    # Mask bit 'j' is set if sum 'j' is achievable. (Indices 1 to 9)
    # The mask array size is 1 << 10, covering bits for sums 0 to 9.
    dp = [0] * (1 << 10)
    dp[0] = 1 # Base case: 0 dice, no sums (1-9) are achievable (empty mask), 1 way.

    for i in range(N):
        current_A = A[i]
        next_dp = [0] * (1 << 10) # DP table for the next iteration

        # Iterate through all current mask states
        for mask in range(1 << 10):
            if dp[mask] == 0:
                continue

            count = dp[mask] # Number of ways to reach this mask with previous dice

            # --- Handle values for the current die from 1 to min(current_A, 10) ---
            # These are the only values that can potentially form sum 10, or contribute
            # to sums less than 10.
            for val in range(1, min(current_A, 10) + 1):
                sum_is_10_found = False

                # Case 1: 'val' itself is 10
                if val == 10:
                    sum_is_10_found = True
                
                # Case 2: 'val' combines with an existing sum in 'mask' to form 10
                # Only check if sum_is_10_found is not already true (from Case 1)
                if not sum_is_10_found:
                    for prev_sum in range(1, 10): # prev_sum ranges from 1 to 9
                        if (mask >> prev_sum) & 1: # If prev_sum is an achievable sum in the current mask
                            if prev_sum + val == 10:
                                sum_is_10_found = True
                                break # Found a sum of 10, this specific outcome satisfies the condition

                if sum_is_10_found:
                    # If a sum of 10 is found, this outcome (current die value + previous configuration)
                    # satisfies the original problem's condition. Thus, it does NOT contribute
                    # to the complement count (next_dp). We skip to the next 'val'.
                    continue

                # If we reach here, this outcome (S_i = val, given previous states in mask)
                # does NOT result in a sum of 10. So, it contributes to next_dp (complement count).
                new_mask = mask
                
                # Add 'val' itself to new_mask if val < 10
                # (Note: val cannot be 10 here, as that would have triggered sum_is_10_found)
                if val < 10:
                    new_mask |= (1 << val)

                # Add combinations of 'val' with 'prev_sum's to new_mask if their sum is < 10
                for prev_sum in range(1, 10):
                    if (mask >> prev_sum) & 1:
                        current_sum = prev_sum + val
                        if current_sum < 10:
                            new_mask |= (1 << current_sum)
                
                # Add the count of ways leading to 'mask' to the new_mask in next_dp
                next_dp[new_mask] = (next_dp[new_mask] + count) % MOD
            
            # --- Handle values for the current die greater than 10 (if any) ---
            # These values cannot form 10 by themselves, nor can they contribute to a sum of 10
            # when combined with other positive numbers (as all numbers are positive).
            # Also, they cannot contribute to sums less than 10.
            # So, for each of these values, the mask simply remains the same.
            remaining_vals_count = max(0, current_A - 10)
            if remaining_vals_count > 0:
                next_dp[mask] = (next_dp[mask] + count * remaining_vals_count) % MOD
        
        # Update dp for the next iteration
        dp = next_dp

    # After processing all N dice, sum up all counts in the final dp table
    # to get the total number of outcomes that satisfy the complement condition.
    total_complement_outcomes = 0
    for val_count in dp:
        total_complement_outcomes = (total_complement_outcomes + val_count) % MOD

    # Calculate total possible outcomes for all dice
    total_outcomes = 1
    for val_A in A:
        total_outcomes = (total_outcomes * val_A) % MOD

    # Calculate modular multiplicative inverse of total_outcomes
    # (Using Fermat's Little Theorem since MOD is a prime number)
    total_outcomes_inv = pow(total_outcomes, MOD - 2, MOD)

    # Probability of the complement event: P(complement) = (complement_outcomes / total_outcomes) % MOD
    prob_complement = (total_complement_outcomes * total_outcomes_inv) % MOD

    # Probability of the original event: P(original) = 1 - P(complement)
    # Ensure result is positive by adding MOD before taking modulo
    ans = (1 - prob_complement + MOD) % MOD
    
    sys.stdout.write(str(ans) + '
')

solve()