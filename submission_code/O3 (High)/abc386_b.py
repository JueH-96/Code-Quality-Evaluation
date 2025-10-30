def main() -> None:
    import sys

    S = sys.stdin.readline().strip()
    n = len(S)

    presses = 0
    i = 0
    while i < n:
        # Use the "00" button whenever we see two consecutive zeros.
        if S[i] == '0' and i + 1 < n and S[i + 1] == '0':
            presses += 1
            i += 2            # one press covers two digits
        else:
            presses += 1
            i += 1            # normal single-digit press

    print(presses)


if __name__ == "__main__":
    main()