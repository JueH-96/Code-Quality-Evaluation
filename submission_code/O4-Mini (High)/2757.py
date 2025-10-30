class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        # Subtract one from a non-negative integer string, preserving length with leading zeros
        def dec_str(s: str) -> str:
            arr = list(s)
            i = len(arr) - 1
            # borrow until we find a non-zero digit
            while i >= 0 and arr[i] == '0':
                arr[i] = '9'
                i -= 1
            if i >= 0:
                # decrement this digit
                arr[i] = chr(ord(arr[i]) - 1)
            return ''.join(arr)

        from functools import lru_cache

        # Count numbers x in [0, bound] whose digit-sum is in [min_sum, max_sum]
        def count_up_to(bound: str) -> int:
            digits = list(map(int, bound))
            n = len(digits)

            @lru_cache(None)
            def dfs(pos: int, tight: bool, s: int) -> int:
                # if digit-sum exceeds max_sum, no valid completions
                if s > max_sum:
                    return 0
                # reached end: check if sum in range
                if pos == n:
                    return 1 if min_sum <= s <= max_sum else 0

                limit = digits[pos] if tight else 9
                res = 0
                for d in range(limit + 1):
                    res += dfs(pos + 1, tight and (d == limit), s + d)
                    if res >= MOD:
                        res %= MOD
                return res

            return dfs(0, True, 0)

        # compute answer = count(0..num2) - count(0..num1-1)
        dec_num1 = dec_str(num1)
        ans = count_up_to(num2) - count_up_to(dec_num1)
        return ans % MOD