class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        import sys
        import threading
        def main():
            MOD = 10 ** 9 + 7
            sys.setrecursionlimit(1 << 25)
            n_bits = len(s)
            s_bits = [int(c) for c in s]

            # Precompute f(m) for m from 1 to 800
            max_m = n_bits  # Maximum number of set bits possible
            f = [-1] * (max_m + 1)

            def count_set_bits(m):
                return bin(m).count('1')

            memo_f = {}

            def compute_f(m):
                if m == 0:
                    return 0
                if m == 1:
                    return 0
                if m in memo_f:
                    return memo_f[m]
                cnt_m = count_set_bits(m)
                memo_f[m] = 1 + compute_f(cnt_m)
                return memo_f[m]

            valid_counts = set()
            for m in range(1, max_m + 1):
                steps = compute_f(m)
                if steps <= k:
                    valid_counts.add(m)

            memo = {}

            from functools import lru_cache

            @lru_cache(maxsize=None)
            def dp(pos, cnt, tight, leading_zero):
                if pos == n_bits:
                    if leading_zero:
                        return 0
                    else:
                        if cnt in valid_counts:
                            return 1
                        else:
                            return 0
                key = (pos, cnt, tight, leading_zero)
                if key in memo:
                    return memo[key]
                res = 0
                max_digit = s_bits[pos] if tight else 1
                for digit in range(0, max_digit + 1):
                    new_leading_zero = leading_zero and (digit == 0)
                    new_cnt = cnt
                    if not new_leading_zero:
                        new_cnt += digit
                    new_tight = tight and (digit == max_digit)
                    res += dp(pos + 1, new_cnt, new_tight, new_leading_zero)
                memo[key] = res % MOD
                return memo[key]

            ans = dp(0, 0, True, True)
            return ans % MOD

        threading.Thread(target=main).start()