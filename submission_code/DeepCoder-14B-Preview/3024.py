MOD = 10**9 + 7

class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        if len(t) != n:
            return 0
        
        # Function to find all valid rotation positions
        def find_valid_rotations(s, t):
            s_doubled = s + s
            valid = []
            start = 0
            while True:
                pos = s_doubled.find(t, start)
                if pos == -1:
                    break
                if pos < len(s):
                    valid.append(pos)
                start = pos + 1
            return valid
        
        valid_r = find_valid_rotations(s, t)
        if not valid_r:
            return 0
        
        # Calculate necessary powers and inverses
        if n == 1:
            pow_nm1_k = 0
        else:
            pow_nm1_k = pow(n-1, k, MOD)
        
        if k % 2 == 0:
            pow_neg1 = 1
        else:
            pow_neg1 = MOD - 1
        
        inv_n = pow(n, MOD-2, MOD)
        
        sum_ways = 0
        for r in valid_r:
            if r == 0:
                term = ((n-1) * pow_neg1 + pow_nm1_k) % MOD
            else:
                term = (pow_nm1_k - pow_neg1) % MOD
            ways_r = (term * inv_n) % MOD
            sum_ways = (sum_ways + ways_r) % MOD
        
        return sum_ways % MOD