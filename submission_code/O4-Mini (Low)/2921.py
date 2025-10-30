import functools

class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7

        def decrement_str_num(s: str) -> str:
            # subtract 1 from the non-negative decimal string s
            lst = list(s)
            i = len(lst) - 1
            while i >= 0:
                if lst[i] == '0':
                    lst[i] = '9'
                    i -= 1
                else:
                    lst[i] = str(ord(lst[i]) - ord('0') - 1)
                    break
            # strip leading zeros
            res = ''.join(lst).lstrip('0')
            return res if res != '' else '0'

        def count_upto(s: str) -> int:
            n = len(s)

            @functools.lru_cache(None)
            def dp(pos: int, prev: int, tight: bool, started: bool) -> int:
                # prev: -1 if no digit started yet, else 0..9
                if pos == n:
                    # count only if we've started a number (non-empty)
                    return 1 if started else 0
                limit = int(s[pos]) if tight else 9
                res = 0
                # try all digits 0..limit
                for d in range(0, limit+1):
                    ntight = tight and (d == limit)
                    if not started:
                        # still skipping leading zeros
                        if d == 0:
                            # skip this digit, still not started
                            res += dp(pos+1, -1, ntight, False)
                        else:
                            # start the number with d
                            res += dp(pos+1, d, ntight, True)
                    else:
                        # already started, check stepping condition
                        if abs(d - prev) == 1:
                            res += dp(pos+1, d, ntight, True)
                        # else invalid, skip
                return res % MOD

            return dp(0, -1, True, False)

        # count stepping numbers in [low, high] = count_upto(high) - count_upto(low-1)
        low_minus_one = decrement_str_num(low)
        ans = (count_upto(high) - count_upto(low_minus_one)) % MOD
        return ans