import sys


def main() -> None:
    # Read input
    data = sys.stdin.read().strip().split()
    n = int(data[0])
    s = data[1]

    # Counts of substrings ending at current position whose value is 0 / 1
    if s[0] == '1':
        c1, c0 = 1, 0          # one substring (the first char) whose value is 1
    else:
        c1, c0 = 0, 1          # one substring (the first char) whose value is 0

    ans = c1                   # total sum after processing the first character

    # Process the rest of the string
    for ch in s[1:]:
        if ch == '0':
            # All previous substrings turn to 1; the new 1-length substring is 0
            c1 = c0 + c1
            c0 = 1
        else:  # ch == '1'
            # Previous 0's turn to 1, previous 1's turn to 0; plus the new 1-length substring (value 1)
            new_c1 = c0 + 1
            new_c0 = c1
            c1, c0 = new_c1, new_c0

        ans += c1              # add the number of ones among substrings ending here

    print(ans)


if __name__ == "__main__":
    main()