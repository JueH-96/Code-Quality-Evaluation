def main():
    import sys

    S = sys.stdin.readline().strip()

    presses = 0
    i = 0
    n = len(S)

    # We'll iterate over S and count consecutive zeros whenever we encounter them.
    # For a run of k zeros, we can use (k // 2) presses of "00" and (k % 2) presses of "0" if there's a leftover zero.
    while i < n:
        if S[i] != '0':
            # Non-zero digit requires exactly one press.
            presses += 1
            i += 1
        else:
            # Count consecutive zeros.
            count_zero = 0
            while i < n and S[i] == '0':
                count_zero += 1
                i += 1
            # Each pair of zeros can be pressed once by "00" button.
            presses += count_zero // 2
            # If there's an odd number of zeros, one zero remains and needs a single "0" press.
            presses += count_zero % 2

    print(presses)

if __name__ == "__main__":
    main()