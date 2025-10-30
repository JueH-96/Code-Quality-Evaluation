def main() -> None:
    import sys

    S = sys.stdin.readline().strip()
    n = len(S)
    ans = 0

    # 0-indexed positions: j is the middle of the triple (must be 'B')
    for j in range(1, n - 1):
        if S[j] != 'B':
            continue
        # maximum possible distance so that i = j-d >= 0 and k = j+d < n
        d_max = min(j, n - j - 1)
        for d in range(1, d_max + 1):
            if S[j - d] == 'A' and S[j + d] == 'C':
                ans += 1

    print(ans)


if __name__ == "__main__":
    main()