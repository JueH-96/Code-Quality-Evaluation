class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def subtract_one(s: str) -> str:
            s_list = list(s)
            i = len(s_list) - 1
            while i >= 0 and s_list[i] == '0':
                s_list[i] = '9'
                i -= 1
            if i < 0:
                return '0'
            s_list[i] = str(int(s_list[i]) - 1)
            res = ''.join(s_list)
            if res[0] == '0':
                first = 0
                while first < len(res) and res[first] == '0':
                    first += 1
                if first == len(res):
                    return '0'
                return res[first:]
            else:
                return res
        
        def count(s: str) -> int:
            digits = list(map(int, s))
            n = len(digits)
            
            from functools import lru_cache
            
            @lru_cache(None)
            def dp(pos, prev, tight, leading):
                if pos == n:
                    return 1
                max_digit = digits[pos] if tight else 9
                total = 0
                for d in range(0, max_digit + 1):
                    new_tight = tight and (d == max_digit)
                    new_leading = leading and (d == 0)
                    if new_leading:
                        total += dp(pos + 1, 10, new_tight, new_leading)
                    else:
                        if leading:
                            total += dp(pos + 1, d, new_tight, False)
                        else:
                            if abs(d - prev) == 1:
                                total += dp(pos + 1, d, new_tight, False)
                            else:
                                continue
                    total %= MOD
                return total % MOD
            
            return dp(0, 10, True, True)
        
        high_count = count(high)
        low_minus_1 = subtract_one(low)
        low_minus_1_count = count(low_minus_1)
        ans = (high_count - low_minus_1_count) % MOD
        return ans