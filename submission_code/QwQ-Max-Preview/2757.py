class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def decrement_str(s):
            s_list = list(s)
            i = len(s) - 1
            while i >= 0 and s_list[i] == '0':
                i -= 1
            if i == -1:
                return '0'  # All zeros, but according to constraints, this won't happen
            s_list[i] = str(int(s_list[i]) - 1)
            for j in range(i + 1, len(s_list)):
                s_list[j] = '9'
            res = ''.join(s_list).lstrip('0')
            return res if res else '0'
        
        def count_up_to(s, min_s, max_s):
            n = len(s)
            digits = list(map(int, s))
            
            from functools import lru_cache
            
            @lru_cache(maxsize=None)
            def dfs(pos, tight, leading_zero, sum_so_far):
                if pos == n:
                    return 1 if (sum_so_far >= min_s and sum_so_far <= max_s) else 0
                res = 0
                max_d = digits[pos] if tight else 9
                for d in range(0, max_d + 1):
                    new_tight = tight and (d == max_d)
                    new_leading_zero = leading_zero and (d == 0)
                    
                    if leading_zero:
                        if d == 0:
                            new_sum = sum_so_far
                        else:
                            new_sum = d
                    else:
                        new_sum = sum_so_far + d
                    
                    if new_sum > max_s:
                        continue  # Prune as sum exceeds max_s
                    
                    res += dfs(pos + 1, new_tight, new_leading_zero, new_sum)
                    res %= MOD
                return res % MOD
            
            return dfs(0, True, True, 0)
        
        upper = count_up_to(num2, min_sum, max_sum)
        lower_str = decrement_str(num1)
        lower = count_up_to(lower_str, min_sum, max_sum)
        
        return (upper - lower) % MOD