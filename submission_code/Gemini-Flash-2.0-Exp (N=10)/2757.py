class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def digit_sum(s):
            return sum(int(digit) for digit in s)

        def count_valid(num_str, min_sum, max_sum, is_upper_bound):
            n = len(num_str)
            memo = {}

            def dp(index, current_sum, is_tight):
                if index == n:
                    return 1 if min_sum <= current_sum <= max_sum else 0
                
                if (index, current_sum, is_tight) in memo:
                    return memo[(index, current_sum, is_tight)]
                
                ans = 0
                upper_limit = int(num_str[index]) if is_tight else 9
                
                for digit in range(upper_limit + 1):
                    ans = (ans + dp(index + 1, current_sum + digit, is_tight and digit == upper_limit)) % MOD
                
                memo[(index, current_sum, is_tight)] = ans
                return ans

            return dp(0, 0, True)

        
        count_num2 = count_valid(num2, min_sum, max_sum, True)
        count_num1_minus_1 = count_valid(str(int(num1) - 1), min_sum, max_sum, True) if int(num1) > 1 else 0
        
        return (count_num2 - count_num1_minus_1 + MOD) % MOD