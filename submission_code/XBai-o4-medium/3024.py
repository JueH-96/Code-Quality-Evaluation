class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        double_s = s + s
        pattern = t
        
        def compute_prefix(pattern):
            m = len(pattern)
            pi = [0] * m
            for i in range(1, m):
                j = pi[i-1]
                while j > 0 and pattern[i] != pattern[j]:
                    j = pi[j-1]
                if pattern[i] == pattern[j]:
                    j += 1
                pi[i] = j
            return pi
        
        pi = compute_prefix(pattern)
        occurrences = []
        j = 0
        m_pattern = len(pattern)
        for i in range(len(double_s)):
            while j > 0 and double_s[i] != pattern[j]:
                j = pi[j-1]
            if double_s[i] == pattern[j]:
                j += 1
            if j == m_pattern:
                occurrences.append(i - m_pattern + 1)
                j = pi[j-1]
        
        valid_rotations = [r for r in occurrences if 0 <= r < n]
        if not valid_rotations:
            return 0
        
        pow_n_minus_1 = pow(n-1, k, MOD)
        if k % 2 == 0:
            sign = 1
        else:
            sign = MOD - 1
        
        inv_n = pow(n, MOD-2, MOD)
        total = 0
        for r in valid_rotations:
            if r == 0:
                numerator = (pow_n_minus_1 + sign * (n - 1)) % MOD
            else:
                numerator = (pow_n_minus_1 - sign) % MOD
            numerator = (numerator + MOD) % MOD  # Ensure non-negative
            term = (numerator * inv_n) % MOD
            total = (total + term) % MOD
        
        return total