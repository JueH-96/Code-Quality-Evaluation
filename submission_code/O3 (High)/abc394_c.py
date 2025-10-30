def main() -> None:
    import sys

    s = sys.stdin.readline().strip()
    n = len(s)
    # Work with a mutable list
    chars = list(s)

    i = 0
    while i < n - 1:
        if chars[i] == 'W' and chars[i + 1] == 'A':
            # Replace the left-most occurring “WA” (at positions i, i+1) with “AC”
            chars[i], chars[i + 1] = 'A', 'C'
            # After the change a new “WA” can only appear starting one position earlier
            if i:
                i -= 1
        else:
            i += 1

    print(''.join(chars))


if __name__ == "__main__":
    main()