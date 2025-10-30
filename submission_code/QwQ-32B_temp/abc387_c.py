import sys
from functools import lru_cache

def get_digits(X):
    return list(map(int, str(X)))

def compute_f(d, X):
    if X < 10:
        return 0
    digits = get_digits(X)
    len_digits = len(digits)
    first_digit_X = digits[0]
    
    if first_digit_X < d:
        max_length = len_digits - 1
        if max_length < 2:
            return 0
        total = 0
        for L in range(2, max_length + 1):
            total += d ** (L - 1)
        return total
    elif first_digit_X > d:
        total = 0
        for L in range(2, len_digits + 1):
            total += d ** (L - 1)
        return total
    else:  # first_digit_X == d
        part1 = 0
        for L in range(2, len_digits):
            part1 += d ** (L - 1)
        
        @lru_cache(maxsize=None)
        def dp(pos, tight):
            if pos == len_digits:
                return 1
            res = 0
            max_digit = digits[pos] if tight else 9
            max_possible = min(max_digit, d - 1)
            for d_i in range(0, max_possible + 1):
                new_tight = tight and (d_i == max_digit)
                res += dp(pos + 1, new_tight)
            return res
        
        part2 = dp(1, True)
        return part1 + part2

def main():
    L, R = map(int, sys.stdin.readline().split())
    total = 0
    for d in range(1, 10):
        count_R = compute_f(d, R)
        count_Lminus1 = compute_f(d, L-1)
        total += (count_R - count_Lminus1)
    print(total)

if __name__ == "__main__":
    main()