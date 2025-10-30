def main() -> None:
    import sys

    K = int(sys.stdin.readline().strip())

    # Generate every non-empty subset of the digits 0 … 9
    # and turn it into a 321-like number (digits in strict
    # descending order).  The subset {0} alone is skipped
    # because it represents the non-positive number 0.
    nums = []
    for mask in range(1, 1 << 10):          # 1 … 1023
        if mask == 1:                       # only {0} → skip
            continue
        n = 0
        for d in range(9, -1, -1):          # from 9 down to 0
            if mask & (1 << d):
                n = n * 10 + d
        nums.append(n)

    nums.sort()                             # only 1022 items
    print(nums[K - 1])


if __name__ == '__main__':
    main()