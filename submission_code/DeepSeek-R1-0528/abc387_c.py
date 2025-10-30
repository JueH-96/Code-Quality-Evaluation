def count_rest(s_bound, max_digit):
    if max_digit < 0:
        return 0
    n = len(s_bound)
    dp0 = 0
    dp1 = 1
    for i in range(n):
        new_dp0 = 0
        new_dp1 = 0
        digit_bound = int(s_bound[i])
        if dp1:
            low = 0
            high = min(digit_bound, max_digit)
            for d in range(low, high + 1):
                if d < digit_bound:
                    new_dp0 += dp1
                else:
                    new_dp1 += dp1
        if dp0:
            choices = max_digit + 1
            new_dp0 += dp0 * choices
        dp0, dp1 = new_dp0, new_dp1
    return dp0 + dp1

def count_up_to(x):
    if x < 10:
        return 0
    s = str(x)
    n = len(s)
    total = 0
    for length in range(2, n):
        for d in range(1, 10):
            total += d ** (length - 1)
    first_digit_bound = int(s[0])
    for d0 in range(1, first_digit_bound):
        total += d0 ** (n - 1)
    rest_str = s[1:]
    if rest_str:
        max_digit_rest = first_digit_bound - 1
        total += count_rest(rest_str, max_digit_rest)
    return total

def main():
    import sys
    data = sys.stdin.read().split()
    L = int(data[0])
    R = int(data[1])
    result = count_up_to(R) - count_up_to(L - 1)
    print(result)

if __name__ == '__main__':
    main()