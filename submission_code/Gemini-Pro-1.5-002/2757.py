class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def count_less_than_or_equal_to(num, min_sum, max_sum):
            n = len(num)
            dp = {}

            def solve(idx, current_sum, tight):
                if idx == n:
                    return min_sum <= current_sum <= max_sum
                
                if (idx, current_sum, tight) in dp:
                    return dp[(idx, current_sum, tight)]

                ans = 0
                upper_bound = int(num[idx]) if tight else 9
                for digit in range(upper_bound + 1):
                    new_tight = tight and (digit == upper_bound)
                    if current_sum + digit <= max_sum:
                        ans = (ans + solve(idx + 1, current_sum + digit, new_tight)) % MOD
                
                dp[(idx, current_sum, tight)] = ans
                return ans

            return solve(0, 0, True)

        count2 = count_less_than_or_equal_to(num2, min_sum, max_sum)
        count1 = count_less_than_or_equal_to(num1, min_sum, max_sum)

        
        def digit_sum(num_str):
            s = 0
            for digit in num_str:
                s += int(digit)
            return s

        num1_sum = digit_sum(num1)
        if min_sum <= num1_sum <= max_sum:
            return (count2 - count1 + 1) % MOD
        else:
            return (count2 - count1) % MOD