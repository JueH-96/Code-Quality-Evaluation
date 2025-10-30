class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def count_up_to(num_str, min_sum, max_sum):
            n = len(num_str)
            # dp[pos][sum][tight] = count of valid numbers
            # pos: current position (0 to n-1)
            # sum: current digit sum
            # tight: whether we're still bounded by num_str
            from functools import lru_cache
            
            @lru_cache(None)
            def dp(pos, current_sum, tight):
                # Base case: we've placed all digits
                if pos == n:
                    return 1 if min_sum <= current_sum <= max_sum else 0
                
                # If current sum already exceeds max_sum, no point continuing
                if current_sum > max_sum:
                    return 0
                
                limit = int(num_str[pos]) if tight else 9
                result = 0
                
                for digit in range(0, limit + 1):
                    new_sum = current_sum + digit
                    new_tight = tight and (digit == limit)
                    result = (result + dp(pos + 1, new_sum, new_tight)) % MOD
                
                return result
            
            return dp(0, 0, True)
        
        # Count numbers <= num2 with digit sum in [min_sum, max_sum]
        count2 = count_up_to(num2, min_sum, max_sum)
        
        # Count numbers < num1 with digit sum in [min_sum, max_sum]
        # We need to subtract 1 from num1 to get numbers < num1
        num1_int = int(num1)
        if num1_int == 0:
            count1 = 0
        else:
            num1_minus_1 = str(num1_int - 1)
            count1 = count_up_to(num1_minus_1, min_sum, max_sum)
        
        return (count2 - count1) % MOD