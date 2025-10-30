class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        mod = 10**9 + 7
        
        def calculate(num, min_sum, max_sum):
            n = len(num)
            dp = {}

            def solve(index, tight, digit_sum):
                if index == n:
                    return 1 if min_sum <= digit_sum <= max_sum else 0
                
                if (index, tight, digit_sum) in dp:
                    return dp[(index, tight, digit_sum)]
                
                ans = 0
                limit = int(num[index]) if tight else 9
                
                for digit in range(limit + 1):
                    new_tight = tight and (digit == limit)
                    new_digit_sum = digit_sum + digit
                    if new_digit_sum <= max_sum:
                        ans = (ans + solve(index + 1, new_tight, new_digit_sum)) % mod
                
                dp[(index, tight, digit_sum)] = ans
                return ans

            return solve(0, True, 0)

        ans = (calculate(num2, min_sum, max_sum) - calculate(num1, min_sum, max_sum) + mod) % mod
        
        digit_sum_num1 = sum(int(digit) for digit in num1)
        if min_sum <= digit_sum_num1 <= max_sum:
            ans = (ans + 1) % mod
        
        return ans