class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        max_m = n  # maximum possible count of set bits for numbers up to s is len(s)
        # Precompute step_counts for 1..max_m
        step_counts = [0] * (max_m + 1)  # step_counts[0] is unused
        step_counts[1] = 0
        for x in range(2, max_m + 1):
            m = bin(x).count('1')
            step_counts[x] = 1 + step_counts[m]
        # Precompute allowed for m in 1..max_m
        allowed = [False] * (max_m + 1)
        for m in range(1, max_m + 1):
            if step_counts[m] <= (k - 1):
                allowed[m] = True
        # Now perform digit DP
        from functools import lru_cache
        digits = list(map(int, s))
        
        @lru_cache(None)
        def dp(pos, count_ones, tight, is_num):
            if pos == len(digits):
                return 1 if (is_num and allowed[count_ones]) else 0
            res = 0
            max_digit = digits[pos] if tight else 1
            for d in [0, 1]:
                if d > max_digit:
                    continue
                new_tight = tight and (d == max_digit)
                new_is_num = is_num or (d == 1)
                new_count = count_ones + (1 if d == 1 else 0)
                res += dp(pos + 1, new_count, new_tight, new_is_num)
                res %= MOD
            return res % MOD
        
        total = dp(0, 0, True, False)
        # Check if s itself is valid
        m_s = s.count('1')
        valid_s = allowed[m_s]
        if valid_s:
            total = (total - 1) % MOD
        return total