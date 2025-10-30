class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        from math import ceil, floor

        s_int = int(s)
        len_s = len(s)
        B = 10 ** len_s

        # Compute k_min and k_max
        k_min = (start - s_int + B -1) // B  # ceil((start - s_int) / B)
        k_min = max(0, k_min)  # Ensure k_min >= 0

        k_max = (finish - s_int) // B  # floor((finish - s_int) / B)
        if k_min > k_max:
            return 0  # No valid k, so no powerful integers

        # Now, we need to count the numbers k in [k_min, k_max] whose digits are all <= limit
        k_min_str = str(k_min)
        k_max_str = str(k_max)
        len_k = max(len(k_min_str), len(k_max_str))
        k_min_str = k_min_str.zfill(len_k)
        k_max_str = k_max_str.zfill(len_k)

        from functools import lru_cache

        @lru_cache(None)
        def dfs(pos, tight_low, tight_high):
            if pos == len_k:
                return 1  # Reached the end, valid number

            res = 0

            low_digit = int(k_min_str[pos]) if tight_low else 0
            high_digit = int(k_max_str[pos]) if tight_high else 9  # Max digit is 9 in k

            for d in range(low_digit, high_digit + 1):
                if d > limit:
                    continue  # Skip digits greater than limit
                next_tight_low = tight_low and (d == int(k_min_str[pos]))
                next_tight_high = tight_high and (d == int(k_max_str[pos]))
                res += dfs(pos + 1, next_tight_low, next_tight_high)
            return res
        
        total_count = dfs(0, True, True)
        return total_count