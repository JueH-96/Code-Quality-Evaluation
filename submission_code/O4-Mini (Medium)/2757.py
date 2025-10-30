from functools import lru_cache

class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        # Digit DP: count numbers <= s with digit sum in [min_sum, max_sum]
        def count_leq(s: str) -> int:
            n = len(s)
            digits = list(map(int, s))

            @lru_cache(None)
            def dp(pos: int, cur_sum: int, tight: bool) -> int:
                # If sum already exceeds max_sum, no need to proceed
                if cur_sum > max_sum:
                    return 0
                # If at end, check if cur_sum in range
                if pos == n:
                    return 1 if min_sum <= cur_sum <= max_sum else 0

                limit = digits[pos] if tight else 9
                res = 0
                for d in range(limit + 1):
                    res += dp(pos + 1,
                              cur_sum + d,
                              tight and (d == limit))
                    if res >= MOD:
                        res %= MOD
                return res

            return dp(0, 0, True)

        # Subtract one from a non-negative decimal string
        def subtract_one(s: str) -> str:
            lst = list(s)
            i = len(lst) - 1
            # borrow until we find a non-zero digit
            while i >= 0 and lst[i] == '0':
                lst[i] = '9'
                i -= 1
            if i >= 0:
                lst[i] = str(int(lst[i]) - 1)
            # Drop leading zeros? Not necessary for DP
            return "".join(lst)

        # Compute result = count_leq(num2) - count_leq(num1 - 1)
        num1_minus = subtract_one(num1)
        ans = count_leq(num2)
        # Only subtract if num1_minus is a valid non-negative
        # For num1="1", num1_minus="0" is fine; DP will return 0 if sum<min_sum
        ans = (ans - count_leq(num1_minus)) % MOD
        return ans