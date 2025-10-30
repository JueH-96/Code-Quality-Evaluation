class Solution:
    mod = 10**9 + 7

    def decrement_string(self, s):
        arr = list(map(int, list(s)))
        n = len(arr)
        i = n - 1
        while i >= 0:
            if arr[i] > 0:
                arr[i] -= 1
                break
            else:
                arr[i] = 9
                i -= 1
        res = ''.join(map(str, arr)).lstrip('0')
        return res if res != "" else "0"

    def count_up_to(self, s):
        n = len(s)
        from functools import lru_cache
        @lru_cache(maxsize=None)
        def dfs(pos, tight, started, last):
            if pos == n:
                return 1 if started else 0
            total = 0
            upper_bound = int(s[pos]) if tight else 9
            for d in range(0, upper_bound + 1):
                new_tight = tight and (d == upper_bound)
                if not started:
                    if d == 0:
                        total = (total + dfs(pos + 1, new_tight, False, -1)) % self.mod
                    else:
                        total = (total + dfs(pos + 1, new_tight, True, d)) % self.mod
                else:
                    if abs(d - last) == 1:
                        total = (total + dfs(pos + 1, new_tight, True, d)) % self.mod
            return total
        return dfs(0, True, False, -1) % self.mod

    def countSteppingNumbers(self, low: str, high: str) -> int:
        low_minus = self.decrement_string(low)
        count_high = self.count_up_to(high)
        count_low_minus = self.count_up_to(low_minus)
        result = (count_high - count_low_minus) % self.mod
        if result < 0:
            result += self.mod
        return result