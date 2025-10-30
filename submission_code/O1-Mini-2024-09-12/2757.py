class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def decrement(num: str) -> str:
            num = list(num)
            i = len(num) - 1
            while i >= 0 and num[i] == '0':
                num[i] = '9'
                i -= 1
            if i == -1:
                return '0'
            num[i] = str(int(num[i]) - 1)
            # Remove leading zeros
            result = ''.join(num).lstrip('0')
            return result if result else '0'

        from functools import lru_cache

        def count_num(num: str) -> int:
            n = len(num)

            @lru_cache(None)
            def dp(pos: int, tight: bool, sum_so_far: int) -> int:
                if pos == n:
                    return 1 if min_sum <= sum_so_far <= max_sum else 0
                limit = int(num[pos]) if tight else 9
                total = 0
                for digit in range(0, limit + 1):
                    new_tight = tight and (digit == limit)
                    new_sum = sum_so_far + digit
                    if new_sum > max_sum:
                        continue
                    total = (total + dp(pos + 1, new_tight, new_sum)) % MOD
                return total

            return dp(0, True, 0)

        num1_prev = decrement(num1)
        count2 = count_num(num2)
        count1 = count_num(num1_prev) if num1_prev != '0' else 0
        return (count2 - count1) % MOD