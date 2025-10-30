def main() -> None:
    import sys

    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    ans = 0          # total strawberries
    run = 0          # current consecutive healthy teeth length

    for ch in S:
        if ch == 'O':
            run += 1
        else:
            ans += run // K
            run = 0
    ans += run // K          # last run (if string ends with 'O')

    print(ans)


if __name__ == "__main__":
    main()