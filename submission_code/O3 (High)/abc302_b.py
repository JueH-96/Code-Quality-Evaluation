import sys

def main() -> None:
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    target = "snuke"
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    for r in range(H):
        for c in range(W):
            if grid[r][c] != 's':
                continue
            for dr, dc in directions:
                cells = []
                for k in range(5):
                    nr = r + dr * k
                    nc = c + dc * k
                    if 0 <= nr < H and 0 <= nc < W:
                        cells.append((nr, nc))
                    else:
                        break
                if len(cells) != 5:
                    continue
                if all(grid[cells[i][0]][cells[i][1]] == target[i] for i in range(5)):
                    for rr, cc in cells:
                        # convert to 1-based indices as required
                        print(rr + 1, cc + 1)
                    return

if __name__ == "__main__":
    main()