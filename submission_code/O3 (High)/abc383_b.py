import sys

def main() -> None:
    H, W, D = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    # Collect coordinates of all floor cells
    floors = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floors.append((i, j))

    n = len(floors)           # number of floor cells (guaranteed ≥ 2)

    # Pre-compute Manhattan distances between every pair of floor cells
    dist = [[0] * n for _ in range(n)]
    for a in range(n):
        x1, y1 = floors[a]
        for b in range(n):
            x2, y2 = floors[b]
            dist[a][b] = abs(x1 - x2) + abs(y1 - y2)

    best = 0
    # Try every unordered pair of humidifier positions
    for i in range(n):
        for j in range(i + 1, n):
            humidified = 0
            # Count floor cells covered by at least one of the two
            for k in range(n):
                if dist[i][k] <= D or dist[j][k] <= D:
                    humidified += 1
            best = max(best, humidified)

    print(best)

if __name__ == "__main__":
    main()