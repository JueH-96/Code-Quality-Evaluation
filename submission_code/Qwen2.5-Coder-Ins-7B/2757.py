class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def digit_sum(n):
            return sum(int(digit) for digit in str(n))

        def dp(index, tight, current_sum):
            if index == len(num):
                return 1 if min_sum <= current_sum <= max_sum else 0
            if (index, tight, current_sum) in memo:
                return memo[(index, tight, current_sum)]
            upper_bound = int(num[index]) if tight else 9
            result = 0
            for digit in range(upper_bound + 1):
                new_tight = tight and digit == upper_bound
                result = (result + dp(index + 1, new_tight, current_sum + digit)) % MOD
            memo[(index, tight, current_sum)] = result
            return result

        memo = {}
        num = num2
        count2 = dp(0, True, 0)
        memo = {}
        num = str(int(num1) - 1)
        count1 = dp(0, True, 0)
        return (count2 - count1 + MOD) % MOD