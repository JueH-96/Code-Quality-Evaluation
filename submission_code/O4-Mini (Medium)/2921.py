from functools import lru_cache

class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def dec_str(s: str) -> str:
            # subtract one from decimal string s, return "0" if result <= 0
            if s == "0":
                return "0"
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
            return res if res else "0"

        def count_upto(s: str) -> int:
            # count stepping numbers in [1, s]
            digits = list(map(int, s))
            n = len(digits)
            
            @lru_cache(None)
            def dp(i: int, prev: int, tight: bool, started: bool) -> int:
                # i: current position [0..n)
                # prev: previous digit (0-9) or -1 if none
                # tight: whether prefix is matching s so far
                # started: whether we've placed a non-leading-zero digit
                if i == n:
                    return 1 if started else 0
                res = 0
                up = digits[i] if tight else 9
                # option: skip this position (place leading zero)
                if not started:
                    # place zero but not start the number
                    res = (res + dp(i+1, prev, tight and up == 0, False)) % MOD
                    # start number here with d from 1..up
                    for d in range(1, up+1):
                        res = (res + dp(i+1, d, tight and d == up, True)) % MOD
                else:
                    # must continue number: digit d with |d-prev| == 1
                    for d in (prev-1, prev+1):
                        if 0 <= d <= up:
                            res = (res + dp(i+1, d, tight and d == up, True)) % MOD
                return res

            return dp(0, -1, True, False)

        low_minus = dec_str(low)
        ans_high = count_upto(high)
        ans_lowm = count_upto(low_minus)
        return (ans_high - ans_lowm) % MOD