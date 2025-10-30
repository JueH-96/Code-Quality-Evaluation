import sys

def solve():
    MOD = 998244353

    N = int(sys.stdin.readline())
    A_list = list(map(int, sys.stdin.readline().split()))

    # dp_curr[mask] stores the number of ways to get 'mask' of sums using dice processed so far.
    # mask is a bitmask where k-th bit (0-indexed) set means sum k+1 is achievable.
    # Max sum is 10, so mask length 10 (bits 0 to 9). Size 1 << 10 (1024).
    dp_curr = [0] * (1 << 10)
    dp_curr[0] = 1 # Base case: 0 dice, 0 sums possible (mask 0), 1 way.

    # Iterate through each die
    for k_idx in range(N): 
        current_A_val = A_list[k_idx]
        dp_next = [0] * (1 << 10) # Initialize dp table for the next state

        # Iterate through all possible masks achieved by previous dice
        for mask_old in range(1 << 10):
            if dp_curr[mask_old] == 0: # If this mask was not achievable, skip
                continue
            
            current_dp_val = dp_curr[mask_old] # Number of ways to get mask_old

            # Case 1: Current die roll x_roll is from 1 up to min(current_A_val, 10)
            # These are the rolls that can potentially form new sums <= 10.
            limit_x_roll = min(current_A_val, 10)
            for x_roll in range(1, limit_x_roll + 1):
                mask_new = mask_old
                
                # Add sum x_roll itself (achieved by choosing only the current die)
                # Sum x_roll corresponds to bit x_roll-1 (0-indexed bit)
                mask_new |= (1 << (x_roll - 1))
                
                # Add sums (s + x_roll), where s is a sum achievable from mask_old.
                # If bit j is set in mask_old, sum j+1 is achievable.
                # The new sum is (j+1) + x_roll. Its bit is ((j+1)+x_roll)-1 = j+x_roll.
                # Shifting mask_old left by x_roll achieves this: bit j moves to j+x_roll.
                # Mask with ((1 << 10) - 1) to keep only bits for sums <= 10.
                shifted_contributions = (mask_old << x_roll) & ((1 << 10) - 1)
                mask_new |= shifted_contributions
                
                dp_next[mask_new] = (dp_next[mask_new] + current_dp_val) % MOD
            
            # Case 2: Current die roll x_roll is > 10 (only if current_A_val > 10)
            # These rolls are too large to form new sums <= 10 by themselves,
            # or by adding to any existing sum s >= 1.
            # So, the mask of achievable sums (<= 10) does not change from mask_old.
            if current_A_val > 10:
                num_large_rolls = current_A_val - 10
                # mask_new is the same as mask_old
                mask_new_for_large_rolls = mask_old 
                term_to_add = (current_dp_val * num_large_rolls) % MOD
                dp_next[mask_new_for_large_rolls] = (dp_next[mask_new_for_large_rolls] + term_to_add) % MOD
        
        # Move to the next state
        dp_curr = dp_next

    # Calculate S_fav: sum of dp_curr[mask] for all masks where sum 10 is achievable.
    # Sum 10 corresponds to bit 9 (0-indexed).
    s_fav = 0
    target_sum_bit_representation = (1 << 9) # Bit for sum 10 (0-indexed bit 9)
    for mask_idx in range(1 << 10):
        if (mask_idx & target_sum_bit_representation): # Check if bit 9 is set
            s_fav = (s_fav + dp_curr[mask_idx]) % MOD
            
    # Calculate total number of outcomes T = product of all A_i
    total_outcomes = 1
    for val_A in A_list:
        total_outcomes = (total_outcomes * val_A) % MOD
        
    # Probability = s_fav * T^(-1) mod MOD
    # T^(-1) = pow(total_outcomes, MOD - 2, MOD) by Fermat's Little Theorem.
    inv_total_outcomes = pow(total_outcomes, MOD - 2, MOD)
    
    probability = (s_fav * inv_total_outcomes) % MOD
    
    sys.stdout.write(str(probability) + "
")

if __name__ == '__main__':
    solve()