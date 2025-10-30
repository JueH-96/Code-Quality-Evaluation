# YOUR CODE HERE
import sys

def solve():
    """
    This function reads the input, solves the problem using dynamic programming,
    and prints the final probability modulo 998244353.
    """
    try:
        # Read input from stdin
        N = int(sys.stdin.readline())
        A = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # This part is for handling potential empty input during local testing,
        # it won't be triggered in a standard contest environment.
        # For submission, it's safe to assume input is always correctly formatted.
        return

    MOD = 998244353
    K = 10
    
    # The mask represents achievable subset sums from 1 to K-1=9.
    # It has K-1 bits, leading to 2^(K-1) states.
    MASK_SIZE = 1 << (K - 1)
    MASK_LIMIT = MASK_SIZE - 1

    # dp[mask] stores the number of ways for dice processed so far
    # to have the set of achievable subset sums < K represented by 'mask',
    # without any subset summing to K.
    dp = [0] * MASK_SIZE
    dp[0] = 1  # Base case: with 0 dice, only sum 0 is achievable (empty set). Mask is 0.
    
    # Precomputing new masks is a small optimization.
    # new_mask_precomputed[d-1][old_mask] stores the resulting mask when adding die roll d.
    new_mask_precomputed = [[0] * MASK_SIZE for _ in range(K - 1)]
    for d in range(1, K):
        d_bit = 1 << (d - 1)
        for mask in range(MASK_SIZE):
            # new_mask = old_mask | {d} | {s+d for s in old_mask}
            # Sums are truncated to be less than K.
            new_mask = mask | d_bit | ((mask << d) & MASK_LIMIT)
            new_mask_precomputed[d - 1][mask] = new_mask

    # Process each die one by one
    for a_i in A:
        new_dp = [0] * MASK_SIZE
        
        for mask in range(MASK_SIZE):
            if dp[mask] == 0:
                continue
            
            val_dp = dp[mask]

            # Case 1: Die roll d > K (i.e., d > 10)
            # These outcomes are never "invalid" for the complementary event.
            # The mask of sums < K remains unchanged.
            if a_i > K:
                count = a_i - K
                new_dp[mask] = (new_dp[mask] + val_dp * count) % MOD
                
            # Case 2: Die roll d <= K (i.e., d from 1 to 10)
            for d in range(1, min(a_i, K) + 1):
                # Check if this roll d is invalid for the complementary event,
                # i.e., it allows forming a subset sum of K.
                is_invalid = False
                if d == K:  # Outcome d=10 itself forms sum K.
                    is_invalid = True
                elif (mask >> (K - d - 1)) & 1:  # If sum K-d was achievable before.
                    is_invalid = True
                
                if not is_invalid:
                    # If d < K, a new mask is formed.
                    if d < K:
                        next_mask = new_mask_precomputed[d - 1][mask]
                    else:
                        # This case (d=K) is always invalid, so this branch is not taken.
                        next_mask = mask
                    
                    new_dp[next_mask] = (new_dp[next_mask] + val_dp) % MOD
        
        dp = new_dp
        
    # Total number of outcomes for the complementary event (no subset sum is K).
    count_complement = sum(dp) % MOD
    
    # Total possible outcomes from throwing all N dice.
    total_outcomes = 1
    for val in A:
        total_outcomes = (total_outcomes * val) % MOD
        
    # Calculate P(complement) = count_complement / total_outcomes.
    # Division is done using modular multiplicative inverse.
    inv_total = pow(total_outcomes, MOD - 2, MOD)
    
    prob_complement = (count_complement * inv_total) % MOD
    
    # The probability of the original event is 1 - P(complement).
    prob_event = (1 - prob_complement + MOD) % MOD
    
    print(prob_event)

solve()