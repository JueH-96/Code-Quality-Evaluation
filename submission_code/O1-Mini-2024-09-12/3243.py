class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        s_num = int(s)
        k = len(s)
        ten_k = 10 ** k

        # Calculate y_min and y_max
        y_min = (start - s_num + ten_k - 1) // ten_k
        y_max = (finish - s_num) // ten_k

        # If y_max is negative, no valid numbers
        if y_max < 0:
            return 0

        # Adjust y_min to be at least 0
        y_min = max(0, y_min)

        from functools import lru_cache

        def count_up_to(N):
            s_N = str(N)
            n = len(s_N)

            @lru_cache(None)
            def dp(pos, tight, started):
                if pos == n:
                    return 1 if started else 1  # Count y=0 as valid
                res = 0
                upper = int(s_N[pos]) if tight else limit
                for d in range(0, upper + 1):
                    if d > limit:
                        continue
                    new_tight = tight and (d == upper)
                    new_started = started or (d != 0)
                    res += dp(pos + 1, new_tight, new_started)
                return res

            return dp(0, True, False)

        count_max = count_up_to(y_max)
        count_min = count_up_to(y_min - 1) if y_min > 0 else 0
        return max(0, count_max - count_min)