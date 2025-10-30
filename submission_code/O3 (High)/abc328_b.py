import sys

def all_same_digits(x: int) -> bool:
    """
    Return True if every digit of x is identical.
    """
    s = str(x)
    return len(set(s)) == 1

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    D = list(map(int, data[1:]))

    ans = 0
    for m in range(1, N + 1):          # month number
        if not all_same_digits(m):
            continue                    # month itself is not a repdigit, skip
        digit = int(str(m)[0])          # common digit of the month (1‥9)

        # possible day numbers consisting only of this digit
        cand_days = [digit]             # one-digit day (d)
        two_digit = digit * 11          # two-digit day (dd)
        if two_digit <= 100:            # days are at most 100
            cand_days.append(two_digit)

        days_in_month = D[m - 1]
        for d in cand_days:
            if d <= days_in_month:
                ans += 1

    print(ans)


if __name__ == "__main__":
    main()