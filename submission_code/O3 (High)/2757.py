class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        # ---------- helpers --------------------------------------------------
        # 1) subtract 1 from decimal string
        def dec_one(s: str) -> str:
            if s == "0":
                return "0"
            lst = list(s)
            i = len(lst) - 1
            while i >= 0 and lst[i] == '0':
                lst[i] = '9'
                i -= 1
            lst[i] = str(int(lst[i]) - 1)
            res = ''.join(lst).lstrip('0')
            return res if res else '0'

        # 2) digit-DP: amount of numbers in [0, bound] whose digit sum is in range
        from functools import lru_cache

        def count_upto(bound: str) -> int:
            digits = list(map(int, bound))
            n = len(digits)

            @lru_cache(maxsize=None)
            def dfs(pos: int, cur_sum: int, tight: int) -> int:
                # cur_sum already exceeds max_sum -> impossible
                if cur_sum > max_sum:
                    return 0
                if pos == n:                         # all digits fixed
                    return 1 if min_sum <= cur_sum <= max_sum else 0

                limit = digits[pos] if tight else 9
                total = 0
                for d in range(limit + 1):
                    total += dfs(pos + 1,
                                 cur_sum + d,
                                 tight and d == limit)
                return total % MOD

            return dfs(0, 0, True)

        # ---------- main -----------------------------------------------------
        ans = (count_upto(num2) - count_upto(dec_one(num1))) % MOD
        return ans