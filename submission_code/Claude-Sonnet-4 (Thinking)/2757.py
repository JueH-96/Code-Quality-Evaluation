class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def decrement(s):
            s = list(s)
            i = len(s) - 1
            while i >= 0 and s[i] == '0':
                s[i] = '9'
                i -= 1
            if i >= 0:
                s[i] = str(int(s[i]) - 1)
            return ''.join(s).lstrip('0') or '0'
        
        def solve(num):
            if not num or num == '0':
                return 1 if min_sum <= 0 <= max_sum else 0
            
            memo = {}
            
            def dp(pos, sum_so_far, tight, started):
                if sum_so_far > max_sum:
                    return 0
                
                if pos == len(num):
                    if started and min_sum <= sum_so_far <= max_sum:
                        return 1
                    elif not started and min_sum <= 0 <= max_sum:
                        return 1
                    return 0
                
                if (pos, sum_so_far, tight, started) in memo:
                    return memo[(pos, sum_so_far, tight, started)]
                
                high = int(num[pos]) if tight else 9
                
                result = 0
                for digit in range(0, high + 1):
                    new_sum = sum_so_far + digit
                    new_tight = tight and (digit == int(num[pos]))
                    new_started = started or (digit > 0)
                    
                    result = (result + dp(pos + 1, new_sum, new_tight, new_started)) % MOD
                
                memo[(pos, sum_so_far, tight, started)] = result
                return result
            
            return dp(0, 0, True, False)
        
        num1_dec = decrement(num1)
        return (solve(num2) - solve(num1_dec)) % MOD