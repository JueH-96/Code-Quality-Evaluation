class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def subtract_one(N_str):
            N_int = int(N_str)
            N_int -= 1
            if N_int < 0:
                return "0"
            return str(N_int)
        
        def count_up_to(N_str):
            from functools import lru_cache
            N = N_str
            @lru_cache(maxsize=None)
            def dp(pos, sum_so_far, tight):
                if pos == len(N):
                    if min_sum <= sum_so_far <= max_sum:
                        return 1
                    else:
                        return 0
                if sum_so_far > max_sum:
                    return 0
                if sum_so_far + (len(N) - pos - 1) * 9 < min_sum:
                    return 0
                upper = int(N[pos]) if tight else 9
                total = 0
                for d in range(0, upper + 1):
                    new_tight = tight and (d == upper)
                    total += dp(pos + 1, sum_so_far + d, new_tight)
                return total
            return dp(0, 0, True)
        
        num1_minus_one = subtract_one(num1)
        result = (count_up_to(num2) - count_up_to(num1_minus_one)) % MOD
        return result