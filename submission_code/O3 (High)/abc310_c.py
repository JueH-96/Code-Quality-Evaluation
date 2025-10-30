def main() -> None:
    import sys

    input = sys.stdin.readline
    N = int(input())
    seen = set()

    for _ in range(N):
        s = input().strip()          # string on the current stick
        r = s[::-1]                  # its reversal
        canonical = s if s < r else r
        seen.add(canonical)

    print(len(seen))


if __name__ == "__main__":
    main()