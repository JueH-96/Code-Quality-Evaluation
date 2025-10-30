def main():
    import sys
    K = int(sys.stdin.readline().strip())

    # We generate all possible strictly decreasing digit sequences
    # by considering all non-empty subsets of {0, 1, 2, ..., 9}.
    # Then we only keep those whose leading (largest) digit is not 0.
    # Finally, we sort them in ascending order and pick the K-th.

    numbers = []
    # 1 << 10 is 2^10 = 1024, so we range from 1..1023 inclusive
    for mask in range(1, 1 << 10):
        digits = []
        for d in range(10):
            if mask & (1 << d):
                digits.append(d)
        # Sort the digits in strictly decreasing order (largest digit first)
        digits.sort(reverse=True)

        # If the largest digit is 0, it's not a valid 321-like number
        if digits[0] == 0:
            continue

        # Convert the digits to an integer
        value = 0
        for d in digits:
            value = value * 10 + d
        numbers.append(value)

    # Sort the collected numbers in ascending order
    numbers.sort()

    # Output the K-th smallest (1-based index => K-1 in 0-based)
    print(numbers[K - 1])

# Do not forget to call main()
main()