mod_val = 10**9 + 7

class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        def subtract_one(s):
            n = list(s)
            i = len(n) - 1
            while i >= 0 and n[i] == '0':
                n[i] = '9'
                i -= 1
            if i < 0:
                return "0"
            n[i] = str(int(n[i]) - 1)
            res = ''.join(n).lstrip('0')
            return res if res != "" else "0"
        
        def count_up_to(s, max_val):
            if max_val < 0:
                return 0
            n_len = len(s)
            from functools import lru_cache
            @lru_cache(maxsize=None)
            def dp(pos, tight, curr_sum):
                if curr_sum > max_val:
                    return 0
                if pos == n_len:
                    return 1
                total = 0
                upper_bound = int(s[pos]) if tight else 9
                for digit in range(0, upper_bound + 1):
                    new_tight = tight and (digit == upper_bound)
                    new_sum = curr_sum + digit
                    total = (total + dp(pos + 1, new_tight, new_sum)) % mod_val
                return total % mod_val
            return dp(0, True, 0) % mod_val
        
        num1_minus_1 = subtract_one(num1)
        A = count_up_to(num2, max_sum)
        B = count_up_to(num2, min_sum - 1)
        C = count_up_to(num1_minus_1, max_sum)
        D = count_up_to(num1_minus_1, min_sum - 1)
        result = (A - B - C + D) % mod_val
        return result