class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        from functools import lru_cache

        def count_upto(s: str) -> int:
            n = len(s)

            @lru_cache(None)
            def dfs(pos: int, last: int, tight: bool, started: bool) -> int:
                # If we've placed all digits, count 1 if we've started a valid number
                if pos == n:
                    return 1 if started else 0

                res = 0
                limit = int(s[pos]) if tight else 9

                for d in range(limit + 1):
                    ntight = tight and (d == limit)
                    if not started:
                        # We haven't started the number yet: we can skip (d=0) or start (d>0)
                        if d == 0:
                            # still not started
                            res = (res + dfs(pos + 1, last, ntight, False)) % MOD
                        else:
                            # start the number with digit d
                            res = (res + dfs(pos + 1, d, ntight, True)) % MOD
                    else:
                        # We have started: adjacent digit must differ by 1
                        if abs(d - last) == 1:
                            res = (res + dfs(pos + 1, d, ntight, True)) % MOD

                return res

            return dfs(0, 0, True, False)

        def dec_str(num: str) -> str:
            # subtract 1 from a decimal string num >= "0"
            arr = list(num)
            i = len(arr) - 1
            while i >= 0 and arr[i] == '0':
                arr[i] = '9'
                i -= 1
            if i >= 0:
                arr[i] = str(int(arr[i]) - 1)
            # strip leading zeros, but leave at least "0"
            s2 = ''.join(arr).lstrip('0')
            return s2 if s2 else '0'

        low_minus_one = dec_str(low)
        ans = count_upto(high) - count_upto(low_minus_one)
        return ans % MOD