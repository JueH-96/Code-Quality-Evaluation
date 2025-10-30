class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        
        # We will build x bit by bit from the most significant down to 0,
        # using a simple branch‐and‐bound greedy: at each bit we pick 0 or 1
        # that gives the larger possible upper bound on the final product.
        
        UA = 0  # partial value of (a xor x) from bits > i
        UB = 0  # partial value of (b xor x) from bits > i
        x = 0
        
        # iterate bits i = n-1 down to 0
        for i in range(n-1, -1, -1):
            ai = (a >> i) & 1
            bi = (b >> i) & 1
            max_low = (1 << i) - 1  # maximum sum from lower i bits
            
            best_choice = 0
            best_bound = -1
            
            # try xi = 0 or 1
            for xi in (0, 1):
                UA2 = UA + ((ai ^ xi) << i)
                UB2 = UB + ((bi ^ xi) << i)
                # best possible if all lower bits turned to 1 for both
                bound = (UA2 + max_low) * (UB2 + max_low)
                if bound > best_bound:
                    best_bound = bound
                    best_choice = xi
            
            # fix the chosen bit
            x |= (best_choice << i)
            UA += ((ai ^ best_choice) << i)
            UB += ((bi ^ best_choice) << i)
        
        result = (UA * UB) % MOD
        return result