import sys

# Pre-compute   S_k = Σ_{d=1..9} d^k   for k = 1 … 18
MAX_EXP = 18          # 10^18 has 19 digits → exponent at most 18
EXP_SUM = [0] * (MAX_EXP + 1)
for k in range(1, MAX_EXP + 1):
    EXP_SUM[k] = sum(pow(d, k) for d in range(1, 10))


def suffix_count(suffix_digits, top):
    """
    suffix_digits : list of integers (each 0–9) – the digits after the top digit
    top           : the most significant digit of the number (1–9)

    Returns the amount of ways to choose the remainder so that
    0 ≤ each chosen digit < top  and  resulting suffix  ≤  given suffix.
    """
    cnt = 0
    m = len(suffix_digits)
    for i, s in enumerate(suffix_digits):
        remaining = m - i - 1          # digits still to be chosen after position i

        if s >= top:
            # every allowed digit (0 … top-1) is < s ⇒ the number is already smaller
            cnt += top * pow(top, remaining)
            break                      # cannot keep equality any further
        else:
            # choose a smaller digit than s
            cnt += s * pow(top, remaining)
            if i == m - 1:             # we matched the suffix completely
                cnt += 1               # include the number exactly equal to the bound
    return cnt


def count_snake(x: int) -> int:
    """
    Returns the amount of Snake numbers in the interval [10, x].
    """
    if x < 10:
        return 0

    s = str(x)
    n = len(s)
    total = 0

    # 1. lengths strictly smaller than n
    for k in range(1, n - 1):          # k = length-1, so length = k+1 ranges 2 … n-1
        total += EXP_SUM[k]

    # 2. numbers with the same length n
    top_digit = int(s[0])
    rem_len = n - 1

    # 2a) top digit smaller than that of x
    for d in range(1, top_digit):
        total += pow(d, rem_len)

    # 2b) top digit equal
    if rem_len:                        # n ≥ 2 always holds here
        suffix_digits = [int(ch) for ch in s[1:]]
        total += suffix_count(suffix_digits, top_digit)

    return total


def main() -> None:
    L, R = map(int, sys.stdin.readline().split())
    ans = count_snake(R) - count_snake(L - 1)
    print(ans)


if __name__ == "__main__":
    main()