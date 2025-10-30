class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def count_valid(num_str):
            n = len(num_str)
            memo = {}
            
            def dp(pos, digit_sum, tight, started):
                if pos == n:
                    return 1 if started and min_sum <= digit_sum <= max_sum else 0
                
                if (pos, digit_sum, tight, started) in memo:
                    return memo[(pos, digit_sum, tight, started)]
                
                limit = int(num_str[pos]) if tight else 9
                result = 0
                
                for digit in range(0, limit + 1):
                    new_tight = tight and (digit == limit)
                    new_started = started or (digit > 0)
                    new_sum = digit_sum + digit
                    
                    # Pruning: if current sum already exceeds max_sum, skip
                    if new_sum <= max_sum:
                        result = (result + dp(pos + 1, new_sum, new_tight, new_started)) % MOD
                
                memo[(pos, digit_sum, tight, started)] = result
                return result
            
            return dp(0, 0, True, False)
        
        def subtract_one(num_str):
            # Convert string to list for easier manipulation
            digits = list(num_str)
            i = len(digits) - 1
            
            # Find the rightmost non-zero digit
            while i >= 0 and digits[i] == '0':
                digits[i] = '9'
                i -= 1
            
            if i >= 0:
                digits[i] = str(int(digits[i]) - 1)
            
            # Remove leading zeros
            result = ''.join(digits).lstrip('0')
            return result if result else '0'
        
        count2 = count_valid(num2)
        num1_minus_1 = subtract_one(num1)
        count1 = count_valid(num1_minus_1) if num1_minus_1 != '0' else 0
        
        return (count2 - count1) % MOD