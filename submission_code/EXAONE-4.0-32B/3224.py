mod = 10**9 + 7

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        segments = []
        factors = []
        
        if sick[0] > 0:
            L_left = sick[0]
            segments.append(L_left)
            factors.append(1)
        
        for i in range(len(sick) - 1):
            L_mid = sick[i+1] - sick[i] - 1
            if L_mid > 0:
                segments.append(L_mid)
                factors.append(pow(2, L_mid - 1, mod))
        
        if sick[-1] < n - 1:
            L_right = n - 1 - sick[-1]
            segments.append(L_right)
            factors.append(1)
        
        total_steps = sum(segments)
        if total_steps == 0:
            return 1
        
        max_val = total_steps
        fact = [1] * (max_val + 1)
        inv_fact = [1] * (max_val + 1)
        
        for i in range(1, max_val + 1):
            fact[i] = fact[i-1] * i % mod
        
        inv_fact[max_val] = pow(fact[max_val], mod - 2, mod)
        for i in range(max_val, 0, -1):
            inv_fact[i-1] = inv_fact[i] * i % mod
        
        numerator = fact[total_steps]
        denominator = 1
        for L in segments:
            denominator = denominator * fact[L] % mod
        
        multinomial_coeff = numerator * pow(denominator, mod - 2, mod) % mod
        
        total_factor = 1
        for f in factors:
            total_factor = total_factor * f % mod
        
        ans = multinomial_coeff * total_factor % mod
        return ans