class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def solve(num_str, k):
            n = len(num_str)
            memo = {}

            def dp(index, even_count, odd_count, remainder, is_tight):
                if index == n:
                    return 1 if even_count == odd_count and remainder == 0 else 0

                if (index, even_count, odd_count, remainder, is_tight) in memo:
                    return memo[(index, even_count, odd_count, remainder, is_tight)]

                upper = int(num_str[index]) if is_tight else 9
                ans = 0
                for digit in range(upper + 1):
                    new_is_tight = is_tight and (digit == upper)
                    new_remainder = (remainder * 10 + digit) % k
                    new_even_count = even_count + (1 if digit % 2 == 0 else 0)
                    new_odd_count = odd_count + (1 if digit % 2 != 0 else 0)

                    ans += dp(index + 1, new_even_count, new_odd_count, new_remainder, new_is_tight)

                memo[(index, even_count, odd_count, remainder, is_tight)] = ans
                return ans

            return dp(0, 0, 0, 0, True)

        return solve(str(high), k) - solve(str(low - 1), k)