class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        from functools import lru_cache

        # Precompute the number of steps to reach 1 for each possible popcount
        def compute_steps(max_c):
            steps = {}
            def helper(c):
                if c == 1:
                    return 0
                if c in steps:
                    return steps[c]
                cnt = bin(c).count('1')
                steps[c] = 1 + helper(cnt)
                return steps[c]
            
            for c in range(1, max_c + 1):
                helper(c)
            return steps

        max_popcount = len(s)
        steps_to_one = compute_steps(max_popcount)

        @lru_cache(maxsize=None)
        def dp(pos, c, tight, leading_zero):
            if pos == len(s):
                if not leading_zero and steps_to_one.get(c, float('inf')) <= k:
                    return 1
                return 0
            total = 0
            limit = int(s[pos]) if tight else 1
            for bit in range(0, limit + 1):
                new_tight = tight and (bit == limit)
                new_leading_zero = leading_zero and (bit == 0)
                new_c = c + bit if not new_leading_zero else 0
                # If leading_zero is True and bit is 0, we don't count the bit
                if new_leading_zero:
                    total += dp(pos + 1, new_c, new_tight, new_leading_zero)
                else:
                    total += dp(pos + 1, new_c, new_tight, new_leading_zero)
                total %= MOD
            return total

        return dp(0, 0, True, True) % MOD