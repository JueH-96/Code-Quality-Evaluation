class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def is_stepping(num_str):
            for i in range(len(num_str) - 1):
                if abs(int(num_str[i]) - int(num_str[i+1])) != 1:
                    return False
            return True
        
        def count_stepping_numbers_up_to(n_str):
            n = len(n_str)
            dp = {}
            
            def dfs(index, prev_digit, tight, leading_zero):
                if index == n:
                    return 1 if not leading_zero else 0
                if (index, prev_digit, tight, leading_zero) in dp:
                    return dp[(index, prev_digit, tight, leading_zero)]
                
                limit = int(n_str[index]) if tight else 9
                total = 0
                
                for digit in range(0, limit + 1):
                    new_tight = tight and (digit == limit)
                    new_leading_zero = leading_zero and (digit == 0)
                    
                    if leading_zero:
                        total += dfs(index + 1, digit, new_tight, new_leading_zero)
                    else:
                        if abs(digit - prev_digit) == 1:
                            total += dfs(index + 1, digit, new_tight, new_leading_zero)
                    
                    total %= MOD
                
                dp[(index, prev_digit, tight, leading_zero)] = total
                return total
            
            return dfs(0, -1, True, True)
        
        low_num = int(low)
        high_num = int(high)
        
        # Count stepping numbers up to high
        count_high = count_stepping_numbers_up_to(high)
        
        # Count stepping numbers up to low-1
        if low_num > 1:
            low_minus_one = str(low_num - 1)
            count_low_minus_one = count_stepping_numbers_up_to(low_minus_one)
        else:
            count_low_minus_one = 0
        
        # The result is count_high - count_low_minus_one
        return (count_high - count_low_minus_one) % MOD