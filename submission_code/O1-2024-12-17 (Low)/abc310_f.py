def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    MOD = 998244353
    
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # We want to count how many ways (outcomes) yield at least one subset summing to 10.
    # Each outcome corresponds to choosing exactly one face from each die (1..A[i]).
    # Then among the chosen faces (r1, r2, ..., rN), we look at all subsets of these N values.
    # If any subset sums to 10, the outcome is "good".
    
    # We will use a dynamic programming approach that keeps track (bitmask) of which sums up to 10
    # are possible from subsets of the chosen faces so far.
    
    # dp[i][mask] will represent the number of outcomes (ways) after processing i dice,
    # whose possible subset sums (0..10) are exactly described by 'mask' 
    # (where mask has the bit s set if sum s is achievable).
    #
    # Transition:
    # - Let M = min(A[i], 10).  (Because values > 10, if chosen in a subset, exceed sum=10 and do not help form sum=10.)
    # - We pick exactly 1 face from die i (which has A[i] faces):
    #   Face in [1..M]:
    #     This can extend sums by v if we include that face in the subset, so new sums for s are s+v (<= 10).
    #       Hence new_mask = old_mask OR (old_mask << v).
    #     Each such v is 1 distinct choice of face.
    #   Face in [M+1 .. A[i]]:
    #     If v > 10, it can't contribute to sum=10. For that subset, we effectively skip adding a new sum
    #     so new_mask = old_mask. The count of such faces is (A[i] - M).
    #
    # Initial dp: dp[0][1<<0] = 1, meaning with zero dice, the only achievable sum is 0 (bit 0 set).
    # We then proceed. In the end, we sum over dp[N][mask] for all masks that have bit 10 set
    # (meaning sum 10 is achievable as a subset).
    
    # Implementation detail: we'll use rolling arrays for dp to avoid large memory use.
    
    # Precompute mod inverse function:
    def modinv(x, m=MOD):
        return pow(x, m-2, m)
    
    # Step 1: compute total number of outcomes = product(A[i]) mod
    total = 1
    for Ai in A:
        total = (total * Ai) % MOD
    
    # DP arrays
    # dp_current[mask] will hold how many outcomes lead to that mask after processing i dice
    dp_current = [0]*(1<<11)
    dp_current[1<<0] = 1  # bit0 => sum=0 is possible
    
    BIT_MASK_0_10 = (1<<11) - 1  # keep bits up to sum=10
    
    # A small helper to shift a mask by v (OR with the original):
    def shift_and_or(mask, v):
        # shift left by v, clip sums above 10
        shifted = (mask << v) & BIT_MASK_0_10
        return mask | shifted
    
    for Ai in A:
        dp_next = [0]*(1<<11)
        # number of small faces (<=10)
        M = min(Ai, 10)
        
        for old_mask in range(1<<11):
            ways = dp_current[old_mask]
            if ways == 0:
                continue
            
            # 1) If we pick a face >10, that doesn't enable new sums up to 10,
            #    so new_mask = old_mask. There are (Ai - M) such faces.
            if Ai > M:
                dp_next[old_mask] = (dp_next[old_mask] + ways * (Ai - M)) % MOD
            
            # 2) For each v in [1..M], new_mask = shift_and_or(old_mask, v)
            #    This accounts for exactly 1 face chosen in that range.
            for v in range(1, M+1):
                new_mask = shift_and_or(old_mask, v)
                dp_next[new_mask] = (dp_next[new_mask] + ways) % MOD
        
        dp_current = dp_next
    
    # Now dp_current[mask] for i=N dice. We sum over all masks that have bit 10 set.
    good = 0
    bit_for_10 = 1 << 10
    for mask in range(1<<11):
        if (mask & bit_for_10) != 0:
            good = (good + dp_current[mask]) % MOD
    
    # Probability = good / total   (mod 998244353)
    # We want to output (good * inv(total)) mod
    inv_total = modinv(total, MOD)
    ans = (good * inv_total) % MOD
    
    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()