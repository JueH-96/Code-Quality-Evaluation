class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        m = len(s)
        s_num = int(s)

        if s_num > finish:
            return 0

        mod = 10 ** m

        # Compute k_min
        if s_num < start:
            numerator = start - s_num
            if numerator <= 0:
                k_min = 0
            else:
                k_min = (numerator + mod - 1) // mod
        else:
            k_min = 0

        # Compute k_max
        numerator = finish - s_num
        if numerator < 0:
            k_max = -1
        else:
            k_max = numerator // mod

        if k_min > k_max:
            return 0

        # Define the helper function
        def helper(n: int) -> int:
            if n < 0:
                return 0
            s = str(n)
            len_s = len(s)

            total = 0
            for l in range(1, len_s):
                if l == 1:
                    total += (limit + 1)
                else:
                    total += limit * (limit + 1) ** (l - 1)

            from functools import lru_cache

            @lru_cache(maxsize=None)
            def dp(pos, tight):
                if pos == len_s:
                    return 1
                res = 0
                max_d = int(s[pos]) if tight else 9
                for d in range(0, 10):
                    if d > limit:
                        continue
                    if pos == 0 and len_s > 1 and d == 0:
                        continue
                    new_tight = tight and (d == max_d)
                    res += dp(pos + 1, new_tight)
                return res

            same_length = dp(0, True)
            total += same_length
            return total

        # Calculate the count
        count = helper(k_max) - helper(k_min - 1)
        return count