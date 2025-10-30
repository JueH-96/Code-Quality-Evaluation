class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Function to find the rotation needed to turn s into t
        def find_rotation(s, t):
            if s == t:
                return 0
            concatenated = s + s
            index = concatenated.find(t)
            if index == -1 or index >= n:
                return -1
            return index % n
        
        r = find_rotation(s, t)
        if r == -1:
            return 0  # Impossible to transform s into t
        
        # Compute (n-1)^k mod MOD
        power_n_minus_1 = pow(n-1, k, MOD)
        
        # Compute (-1)^k mod MOD
        power_neg1_k = -1 if k % 2 else 1
        power_neg1_k %= MOD
        
        # Compute numerator and denominator
        if r == 0:
            numerator = (power_n_minus_1 - power_neg1_k) % MOD
        else:
            numerator = (power_n_minus_1 + power_neg1_k) % MOD
        denominator = n
        
        # Compute modular inverse of denominator modulo MOD
        inv_denominator = pow(denominator, MOD-2, MOD)
        
        # Compute the final answer
        answer = (numerator * inv_denominator) % MOD
        return answer