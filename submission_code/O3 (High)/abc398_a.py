def main() -> None:
    import sys

    N = int(sys.stdin.readline().strip())

    # start with all '-'
    s = ['-'] * N

    if N % 2:                # odd length → single '=' at the centre
        s[N // 2] = '='
    else:                    # even length → two '=' centred and adjacent
        centre = N // 2
        s[centre - 1] = s[centre] = '='

    print(''.join(s))


if __name__ == "__main__":
    main()