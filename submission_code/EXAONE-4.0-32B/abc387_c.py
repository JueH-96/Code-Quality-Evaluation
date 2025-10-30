import sys

def count_snake(n):
    if n < 10:
        return 0
    s = str(n)
    L = len(s)
    total = 0
    for l in range(2, L):
        for d in range(1, 10):
            total += d ** (l - 1)
    
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def dp(pos, tight, first_digit):
        if pos == L:
            return 1
        total_here = 0
        current_bound = int(s[pos]) if tight else 9
        ub = min(current_bound, first_digit - 1)
        low = 0
        if ub < low:
            return 0
        for d in range(low, ub + 1):
            new_tight = tight and (d == current_bound)
            total_here += dp(pos + 1, new_tight, first_digit)
        return total_here

    first_digit_bound = int(s[0])
    for d0 in range(1, first_digit_bound + 1):
        tight_next = 1 if (d0 == first_digit_bound) else 0
        total += dp(1, tight_next, d0)
    
    return total

def main():
    data = sys.stdin.read().split()
    L_val = int(data[0])
    R_val = int(data[1])
    res = count_snake(R_val) - count_snake(L_val - 1)
    print(res)

if __name__ == "__main__":
    main()