import typing
from typing import *

class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 1000000007
        
        def count_up_to(S: str) -> int:
            L = len(S)
            memo = {}
            
            def dp(pos: int, curr_sum: int, tight: int) -> int:
                if pos == L:
                    return 1 if min_sum <= curr_sum <= max_sum else 0
                if curr_sum > max_sum:
                    return 0
                max_possible_sum = curr_sum + 9 * (L - pos)
                if max_possible_sum < min_sum:
                    return 0
                key = (pos, curr_sum, tight)
                if key in memo:
                    return memo[key]
                ans = 0
                upper_limit = 9 if tight == 0 else int(S[pos])
                for d in range(0, upper_limit + 1):
                    new_tight = 1 if tight == 1 and d == int(S[pos]) else 0
                    new_sum = curr_sum + d
                    ans += dp(pos + 1, new_sum, new_tight)
                    ans %= MOD
                memo[key] = ans
                return memo[key]
            
            return dp(0, 0, 1)
        
        prev_str = str(int(num1) - 1)
        count_num2 = count_up_to(num2)
        count_prev = count_up_to(prev_str)
        result = (count_num2 - count_prev + MOD) % MOD
        return result