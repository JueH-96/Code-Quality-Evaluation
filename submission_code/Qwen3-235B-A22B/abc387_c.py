import sys
from functools import lru_cache

def main():
    L, R = map(int, sys.stdin.readline().split())

    def compute_f(X):
        if X < 10:
            return 0
        s = str(X)
        len_x = len(s)
        sum_l = 0
        # Sum snake numbers for lengths 2 <= l < len_x
        for l in range(2, len_x):
            current = 0
            for d in range(1, 10):
                current += pow(d, l-1)
            sum_l += current
        # Compute count_len_x for X's length
        X_digits = list(map(int, s))
        n = len(X_digits)
        
        @lru_cache(maxsize=None)
        def dp(pos, tight, d0):
            if pos == n:
                return 1
            res = 0
            max_p = X_digits[pos] if tight else 9
            if pos == 0:
                # First digit: 1 to max_p inclusive
                for c in range(1, max_p + 1):
                    new_tight = tight and (c == X_digits[pos])
                    res += dp(pos + 1, new_tight, c)
            else:
                # Subsequent digits must be <= d0 - 1
                upper = min(max_p, d0 - 1)
                if upper >= 0:
                    for c in range(0, upper + 1):
                        new_tight = tight and (c == X_digits[pos])
                        res += dp(pos + 1, new_tight, d0)
            return res
        
        count_len_x = dp(0, True, 0)
        dp.cache_clear()
        return sum_l + count_len_x

    result = compute_f(R) - compute_f(L - 1)
    print(result)

if __name__ == "__main__":
    main()