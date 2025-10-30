def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    target = "snuke"

    for r in range(h):
        for c in range(w):
            if grid[r][c] == 's':
                for dr, dc in directions:
                    path = [(r, c)]
                    current_r, current_c = r, c
                    match = True
                    for i in range(1, 5):
                        current_r += dr
                        current_c += dc
                        if 0 <= current_r < h and 0 <= current_c < w and grid[current_r][current_c] == target[i]:
                            path.append((current_r, current_c))
                        else:
                            match = False
                            break
                    if match:
                        for row_idx, col_idx in path:
                            print(row_idx + 1, col_idx + 1)
                        return

solve()