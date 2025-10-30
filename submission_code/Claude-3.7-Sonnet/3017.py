class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count_up_to(num):
            num_str = str(num)
            n = len(num_str)
            
            # Memoization
            memo = {}
            
            def dp(idx, diff, tight, mod, started):
                if idx == n:
                    # If we've processed all digits, check if conditions are met
                    # Number is beautiful if diff=0 (equal even/odd digits), 
                    # mod=0 (divisible by k), and we've started a number
                    return 1 if diff == 0 and mod == 0 and started else 0
                
                key = (idx, diff, tight, mod, started)
                if key in memo:
                    return memo[key]
                
                limit = int(num_str[idx]) if tight else 9
                res = 0
                
                # If we haven't started a number yet, we can skip this position
                if not started:
                    res += dp(idx + 1, diff, tight and limit == 0, 0, False)
                
                # Determine valid digits we can place
                start_digit = 1 if not started else 0
                
                for digit in range(start_digit, limit + 1):
                    # Calculate the new mod
                    new_mod = (mod * 10 + digit) % k
                    
                    # Calculate new tight constraint
                    new_tight = tight and digit == limit
                    
                    # Update the even/odd digit difference
                    if digit % 2 == 0:
                        res += dp(idx + 1, diff + 1, new_tight, new_mod, True)
                    else:
                        res += dp(idx + 1, diff - 1, new_tight, new_mod, True)
                
                memo[key] = res
                return res
            
            return dp(0, 0, True, 0, False)
        
        # Count beautiful integers in range [low, high]
        return count_up_to(high) - count_up_to(low - 1)