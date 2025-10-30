class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n_str = s
        n_len = len(n_str)
        
        # Precompute dp[m] for m from 1 to 800
        from functools import lru_cache
        @lru_cache(maxsize=None)
        def compute_dp(m):
            if m == 1:
                return 0
            return 1 + compute_dp(bin(m).count('1'))
        
        # Function to count numbers < n with exactly m set bits
        def count_set_bits_less_than_n(m):
            @lru_cache(maxsize=None)
            def dp(pos, tight, count):
                if pos == n_len:
                    return 1 if count == m else 0
                limit = int(n_str[pos]) if tight else 1
                total = 0
                for bit in range(0, limit + 1):
                    new_tight = tight and (bit == limit)
                    new_count = count + bit
                    total += dp(pos + 1, new_tight, new_count)
                return total
            return dp(0, 1, 0)
        
        # Special case when n = 1
        if n_len == 1 and n_str == "1":
            return 0
        
        # Precompute dp[m] for m from 1 to 800
        dp_m = {}
        for m in range(1, 801):
            dp_m[m] = compute_dp(m)
        
        # Sum counts for m where dp[m] + 1 <= k
        answer = 0
        for m in range(1, 801):
            if dp_m[m] + 1 <= k:
                answer = (answer + count_set_bits_less_than_n(m)) % MOD
        
        return answer