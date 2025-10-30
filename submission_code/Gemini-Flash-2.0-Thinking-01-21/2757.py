class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        def calculate_count(n_str, min_sum_val, max_sum_val):
            if int(n_str) < 0:
                return 0
            n_digits = [int(d) for d in n_str]
            n_len = len(n_digits)
            memo = {}
            
            def solve(index, current_sum, tight):
                if current_sum > max_sum_val:
                    return 0
                if index == n_len:
                    return 1 if min_sum_val <= current_sum <= max_sum_val else 0
                if (index, current_sum, tight) in memo:
                    return memo[(index, current_sum, tight)]
                
                count = 0
                limit = n_digits[index] if tight else 9
                for digit in range(limit + 1):
                    next_tight = tight and (digit == limit)
                    count = (count + solve(index + 1, current_sum + digit, next_tight)) % MOD
                    
                memo[(index, current_sum, tight)] = count
                return count
                
            return solve(0, 0, True)
            
        count2 = calculate_count(num2, min_sum, max_sum)
        count1 = calculate_count(str(max(0, int(num1) - 1)), min_sum, max_sum)
        
        result = (count2 - count1 + MOD) % MOD
        return result