class Solution:
    mod = 10**9 + 7

    def countSteppingNumbers(self, low: str, high: str) -> int:
        low_minus = self.subtract_one(low)
        total_high = self.f(high)
        total_low_minus = self.f(low_minus)
        ans = (total_high - total_low_minus) % self.mod
        if ans < 0:
            ans += self.mod
        return ans

    def subtract_one(self, s):
        arr = [int(c) for c in s]
        n = len(arr)
        i = n - 1
        while i >= 0:
            if arr[i] != 0:
                arr[i] -= 1
                break
            else:
                arr[i] = 9
                i -= 1
        s_res = ''.join(str(x) for x in arr).lstrip('0')
        if s_res == '':
            return "0"
        return s_res

    def f(self, s):
        if s == "0":
            return 0
        n = len(s)
        total = 0
        for L in range(1, n):
            total = (total + self.helper(L)) % self.mod
        total = (total + self.helper_with_bound(s)) % self.mod
        return total

    def helper(self, L):
        if L == 0:
            return 0
        from functools import lru_cache
        @lru_cache(maxsize=None)
        def dp(pos, last):
            if pos == L:
                return 1
            total_val = 0
            if pos == 0:
                for d in range(1, 10):
                    total_val = (total_val + dp(pos + 1, d)) % self.mod
            else:
                for d in range(0, 10):
                    if abs(d - last) == 1:
                        total_val = (total_val + dp(pos + 1, d)) % self.mod
            return total_val
        return dp(0, 0)

    def helper_with_bound(self, s):
        n = len(s)
        from functools import lru_cache
        @lru_cache(maxsize=None)
        def dp(pos, tight, last):
            if pos == n:
                return 1
            total_val = 0
            upper = int(s[pos]) if tight else 9
            if pos == 0:
                for d in range(1, upper + 1):
                    new_tight = tight and (d == upper)
                    total_val = (total_val + dp(pos + 1, new_tight, d)) % self.mod
            else:
                for d in range(0, upper + 1):
                    if abs(d - last) == 1:
                        new_tight = tight and (d == upper)
                        total_val = (total_val + dp(pos + 1, new_tight, d)) % self.mod
            return total_val
        return dp(0, True, 0)