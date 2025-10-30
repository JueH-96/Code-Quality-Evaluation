def main():
    import sys
    from functools import lru_cache
    L, R = map(int, sys.stdin.read().split())
    
    def count_up_to(x):
        if x < 10:
            return 0
        s = str(x)
        D = len(s)
        total = 0
        # Add numbers with digit lengths from 2 to D-1
        for m in range(2, D):
            for d in range(1, 10):
                total += d ** (m-1)
        # Handle numbers with D digits
        first_digit = int(s[0])
        rest_str = s[1:]
        rest_digits = list(map(int, rest_str))
        for d in range(1, 10):
            if d < first_digit:
                total += d ** (D-1)
            elif d == first_digit:
                n = len(rest_digits)
                max_digit_allowed = d - 1
                # Compute count_valid using DP
                @lru_cache(maxsize=None)
                def dp(pos, is_tight):
                    if pos == n:
                        return 1
                    res = 0
                    upper = max_digit_allowed
                    if is_tight:
                        upper = min(upper, rest_digits[pos])
                    for digit in range(0, upper + 1):
                        new_is_tight = is_tight and (digit == rest_digits[pos])
                        res += dp(pos + 1, new_is_tight)
                    return res
                cnt = dp(0, True)
                total += cnt
        return total
    
    ans = count_up_to(R) - count_up_to(L - 1)
    print(ans)

if __name__ == "__main__":
    main()