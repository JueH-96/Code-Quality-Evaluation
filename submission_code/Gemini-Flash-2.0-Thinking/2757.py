class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def count_upto(num_str, min_sum, max_sum):
            n = len(num_str)
            memo = {}

            def dp(index, current_sum, is_tight):
                if index == n:
                    return 1 if min_sum <= current_sum <= max_sum else 0

                if (index, current_sum, is_tight) in memo:
                    return memo[(index, current_sum, is_tight)]

                upper_bound = int(num_str[index]) if is_tight else 9
                ans = 0
                for digit in range(upper_bound + 1):
                    if current_sum + digit <= max_sum:
                        new_is_tight = is_tight and (digit == upper_bound)
                        ans = (ans + dp(index + 1, current_sum + digit, new_is_tight)) % MOD

                memo[(index, current_sum, is_tight)] = ans
                return ans

            return dp(0, 0, True)

        count_num2 = count_upto(num2, min_sum, max_sum)

        num1_minus_1 = str(int(num1) - 1)
        count_num1_minus_1 = 0
        if int(num1) > 0:
            count_num1_minus_1 = count_upto(num1_minus_1, min_sum, max_sum)

        result = (count_num2 - count_num1_minus_1 + MOD) % MOD
        return result