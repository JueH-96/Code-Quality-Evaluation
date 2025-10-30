def main():
    import sys
    input = sys.stdin.readline

    H, W = map(int, input().split())
    Si, Sj = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]
    X = input().strip()

    r, c = Si, Sj
    for ch in X:
        nr, nc = r, c
        if ch == 'L':
            nc -= 1
        elif ch == 'R':
            nc += 1
        elif ch == 'U':
            nr -= 1
        elif ch == 'D':
            nr += 1

        if 1 <= nr <= H and 1 <= nc <= W and grid[nr-1][nc-1] == '.':
            r, c = nr, nc

    print(r, c)

if __name__ == "__main__":
    main()