class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count_beautiful(n):
            @lru_cache(None)
            def dfs(pos, even, odd, remainder, tight, leading_zero):
                if pos == len(s):
                    return int(even == odd and remainder == 0)
                res = 0
                limit = int(s[pos]) if tight else 9
                for d in range(0 if not leading_zero else 0, limit + 1):
                    new_even = even + 1 if d % 2 == 0 else even
                    new_odd = odd + 1 if d % 2 == 1 else odd
                    new_remainder = (remainder * 10 + d) % k
                    new_tight = tight and d == int(s[pos])
                    new_leading_zero = leading_zero and d == 0
                    res += dfs(pos + 1, new_even, new_odd, new_remainder, new_tight, new_leading_zero)
                return res

            s = str(n)
            return dfs(0, 0, 0, 0, True, True)

        return count_beautiful(high) - count_beautiful(low - 1)