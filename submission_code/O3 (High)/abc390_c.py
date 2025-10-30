def main() -> None:
    import sys

    input = sys.stdin.readline

    H, W = map(int, input().split())
    S = [input().strip() for _ in range(H)]

    # find the minimum rectangle that contains every '#'
    r_min, r_max = H, -1
    c_min, c_max = W, -1
    for i in range(H):
        for j, ch in enumerate(S[i]):
            if ch == '#':
                r_min = min(r_min, i)
                r_max = max(r_max, i)
                c_min = min(c_min, j)
                c_max = max(c_max, j)

    # inside that rectangle no cell may be '.'
    for i in range(r_min, r_max + 1):
        row = S[i]
        for j in range(c_min, c_max + 1):
            if row[j] == '.':
                print("No")
                return

    print("Yes")


if __name__ == "__main__":
    main()