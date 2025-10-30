def main():
    import sys
    data = sys.stdin.read().splitlines()
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1 + H):
        grid.append(data[i].strip())
    
    directions = [
        (0, 1),   # right
        (1, 0),   # down
        (0, -1),  # left
        (-1, 0),  # up
        (1, 1),   # down-right
        (1, -1),  # down-left
        (-1, 1),  # up-right
        (-1, -1)  # up-left
    ]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 's':
                for dx, dy in directions:
                    r, c = i, j
                    path = []
                    valid = True
                    for idx, target_char in enumerate("snuke"):
                        if r < 0 or r >= H or c < 0 or c >= W:
                            valid = False
                            break
                        if grid[r][c] != target_char:
                            valid = False
                            break
                        path.append((r, c))
                        if idx < 4:
                            r += dx
                            c += dy
                    if valid:
                        for (r, c) in path:
                            print(f"{r+1} {c+1}")
                        return

if __name__ == "__main__":
    main()