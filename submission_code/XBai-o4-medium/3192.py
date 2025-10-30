class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        if n == 0:
            return ((a % MOD) * (b % MOD)) % MOD
        
        a_val = a
        b_val = b
        
        for i in reversed(range(n)):
            delta = 1 << i
            mask = (1 << i) - 1
            # Calculate potential without flipping
            no_flip = (a_val | mask) * (b_val | mask)
            # Calculate potential with flipping
            a_flip = a_val ^ delta
            b_flip = b_val ^ delta
            flip = (a_flip | mask) * (b_flip | mask)
            
            if flip > no_flip:
                a_val, b_val = a_flip, b_flip
            elif flip == no_flip:
                current_diff = abs(a_val - b_val)
                new_diff = abs(a_flip - b_flip)
                if new_diff < current_diff:
                    a_val, b_val = a_flip, b_flip
        
        return ((a_val % MOD) * (b_val % MOD)) % MOD