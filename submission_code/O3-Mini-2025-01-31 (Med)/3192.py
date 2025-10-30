class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        # U = 2^n 
        U = 1 << n
        # Split a and b into a_high (bits beyond n) and a_low (the last n bits)
        a_high, a_low = divmod(a, U)
        b_high, b_low = divmod(b, U)
        
        # For bit positions 0 ... n-1, we “force” the bit decision when a and b are equal.
        # And if they differ, we have a freedom (tie bit) where the two choices
        # yield either (0, 2^i) or (2^i, 0) for (a XOR x, b XOR x) respectively.
        forced = 0       # sum of contributions from positions where a and b agree (always yielding a 1)
        T_sum = 0        # total sum available from tie bits
        tie_values = []  # list of available tie-bit amounts (each is 2^i)
        
        for i in range(n):
            bit_a = (a_low >> i) & 1
            bit_b = (b_low >> i) & 1
            if bit_a == bit_b:
                # We force x_i so that (bit XOR x_i) becomes 1. (If bit is 0, choose x_i = 1; if bit is 1, choose x_i = 0)
                forced += (1 << i)
            else:
                T_sum += (1 << i)
                tie_values.append(1 << i)
                
        # sort tie_values in descending order (largest power first)
        tie_values.sort(reverse=True)
        
        # Let L and R be the “base” parts coming from the higher bits and the forced lower bits:
        L = a_high * U + forced
        R = b_high * U + forced
        # In the continuous (relaxed) version the optimum allocation (for the lower n bits) would be:
        target = ( (R - L) + T_sum ) / 2.0
        # Clamp target into the feasible range: S must be in [0, T_sum].
        if target < 0:
            target = 0
        if target > T_sum:
            target = T_sum
        
        # Now we want to choose some S (achievable as a subset sum from tie_values) which is as close as possible to target.
        # Because the items are powers of 2 and (usually) few (at most n, and n ≤ 50), we can use a greedy selection:
        best_candidate = 0
        overshoot = None
        cur = 0
        for v in tie_values:
            if cur + v <= target:
                cur += v
            else:
                overshoot = cur + v
                break
        cand1 = cur
        if overshoot is None:
            cand2 = cand1
        else:
            cand2 = overshoot
        # choose that candidate which is closer to the target
        if abs(cand1 - target) <= abs(cand2 - target):
            best_S = cand1
        else:
            best_S = cand2
        
        # With the optimal allocation S (for the tie bits going into (a XOR x)),
        # the lower–n–bit contributions become:
        f_low = forced + best_S
        g_low = forced + (T_sum - best_S)
        # And the total numbers are:
        f_total = a_high * U + f_low
        g_total = b_high * U + g_low
        
        return (f_total % MOD) * (g_total % MOD) % MOD

if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.maximumXorProduct(12, 5, 4))   # Expected output: 98
    # Example 2:
    print(sol.maximumXorProduct(6, 7, 5))    # Expected output: 930
    # Example 3:
    print(sol.maximumXorProduct(1, 6, 3))    # Expected output: 12
    
    # Additional tests:
    # When a == b (then there are no tie bits and our forced decision is unique)
    print(sol.maximumXorProduct(2, 2, 4))    # x is forced; result equals (a XOR x)^2.
    # When n == 0, x is forced to be 0 and answer is just a*b.
    print(sol.maximumXorProduct(123, 456, 0))  # Expected output: 123*456  mod (10^9+7)