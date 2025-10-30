class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        if t == 0:
            return len(s) % MOD
        
        # Precompute dp for 'z' transformations up to t
        dp = [0] * (t + 1)
        dp[0] = 1
        for i in range(1, t + 1):
            # Calculate a_part: contribution from 'a' after (i-1) steps
            a_steps = i - 1
            if a_steps <= 25:
                a_part = 1
            else:
                a_part = dp[a_steps - 25] if (a_steps - 25) >= 0 else 0
            
            # Calculate b_part: contribution from 'b' after (i-1) steps
            b_steps = i - 1
            if b_steps <= 24:
                b_part = 1
            else:
                b_part = dp[b_steps - 24] if (b_steps - 24) >= 0 else 0
            
            dp[i] = (a_part + b_part) % MOD
        
        total = 0
        for c in s:
            current = ord(c) - ord('a')
            k = 25 - current
            if t <= k:
                total = (total + 1) % MOD
            else:
                t_prime = t - k
                total = (total + dp[t_prime]) % MOD
        
        return total % MOD