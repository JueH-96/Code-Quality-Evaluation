class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def subtract_one(s):
            s_list = list(s)
            i = len(s_list) - 1
            while i >= 0 and s_list[i] == '0':
                s_list[i] = '9'
                i -= 1
            if i < 0:
                return '0'
            s_list[i] = str(int(s_list[i]) - 1)
            while len(s_list) > 1 and s_list[0] == '0':
                s_list.pop(0)
            return ''.join(s_list)
        
        def count_less_equal(s):
            n = len(s)
            s_digits = [int(c) for c in s]
            
            from functools import lru_cache
            
            @lru_cache(maxsize=None)
            def dp(pos, tight, sum_so_far):
                if pos == n:
                    return 1 if (min_sum <= sum_so_far <= max_sum) else 0
                res = 0
                max_d = s_digits[pos] if tight else 9
                for d in range(0, max_d + 1):
                    new_tight = tight and (d == max_d)
                    res += dp(pos + 1, new_tight, sum_so_far + d)
                    res %= MOD
                return res
            return dp(0, True, 0)
        
        num1_minus_1 = subtract_one(num1)
        upper_num2 = count_less_equal(num2)
        upper_num1_minus_1 = count_less_equal(num1_minus_1)
        return (upper_num2 - upper_num1_minus_1) % MOD