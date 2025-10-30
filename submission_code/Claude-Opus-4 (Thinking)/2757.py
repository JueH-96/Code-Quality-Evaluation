class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def count_up_to(num_str):
            from functools import cache
            
            @cache
            def dp(pos, digit_sum, is_tight, is_num):
                if pos == len(num_str):
                    return int(is_num and min_sum <= digit_sum <= max_sum)
                
                if digit_sum > max_sum:
                    return 0
                
                limit = int(num_str[pos]) if is_tight else 9
                result = 0
                
                # Option to skip this position (leading zero) if we haven't started the number yet
                if not is_num:
                    result = dp(pos + 1, 0, False, False)
                
                # Try placing digits
                start = 1 if not is_num else 0
                for digit in range(start, limit + 1):
                    result = (result + dp(pos + 1, digit_sum + digit, is_tight and digit == limit, True)) % MOD
                
                return result
            
            return dp(0, 0, True, False)
        
        # Count numbers from 1 to num2 with digit sum in [min_sum, max_sum]
        count2 = count_up_to(num2)
        
        # Count numbers from 1 to num1-1 with digit sum in [min_sum, max_sum]
        num1_int = int(num1)
        if num1_int == 1:
            count1 = 0
        else:
            count1 = count_up_to(str(num1_int - 1))
        
        return (count2 - count1) % MOD