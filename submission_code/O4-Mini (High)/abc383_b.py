def main():
    import sys
    input = sys.stdin.readline

    H, W, D = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]

    # Collect all floor cells
    floors = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floors.append((i, j))

    n = len(floors)
    ans = 0

    # Try all pairs of distinct floor cells for humidifier placement
    for a in range(n):
        for b in range(a + 1, n):
            (i1, j1) = floors[a]
            (i2, j2) = floors[b]
            cnt = 0
            # Count how many floor cells are within distance D of at least one humidifier
            for (i, j) in floors:
                if abs(i - i1) + abs(j - j1) <= D or abs(i - i2) + abs(j - j2) <= D:
                    cnt += 1
            if cnt > ans:
                ans = cnt

    print(ans)

if __name__ == "__main__":
    main()