class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10 ** 9 + 7

        def count(N_str):
            memo = {}
            n = len(N_str)

            def dp(pos, prev_digit, tight, leading_zero):
                if pos == n:
                    if not leading_zero:
                        return 1
                    else:
                        return 0

                key = (pos, prev_digit, tight, leading_zero)
                if key in memo:
                    return memo[key]

                res = 0
                limit = int(N_str[pos]) if tight else 9
                if leading_zero:
                    for d in range(1, limit+1):
                        new_tight = tight and (d == limit)
                        res += dp(pos+1, d, new_tight, False)
                else:
                    for d in range(0, limit+1):
                        if abs(d - prev_digit) != 1:
                            continue
                        new_tight = tight and (d == limit)
                        res += dp(pos+1, d, new_tight, False)
                memo[key] = res % MOD
                return res % MOD

            return dp(0, -1, True, True)

        def subtract_one(s):
            if s == '0':
                return s
            elif s == '1':
                return '0'
            else:
                s = list(s)
                i = len(s) - 1
                while s[i] == '0':
                    s[i] = '9'
                    i -= 1
                s[i] = str(int(s[i]) - 1)
                while len(s) > 1 and s[0] == '0':
                    s.pop(0)
                return ''.join(s)

        low_minus_one = subtract_one(low)
        total = (count(high) - count(low_minus_one)) % MOD
        return total