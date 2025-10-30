def main():
    import sys
    data = sys.stdin.read().splitlines()
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    target = "snuke"
    dirs = [(-1,-1), (-1,0), (-1,1),
            (0,-1),           (0,1),
            (1,-1),  (1,0),   (1,1)]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == target[0]:
                for dr, dc in dirs:
                    positions = [(i, j)]
                    valid = True
                    for step in range(1, 5):
                        ni = i + step * dr
                        nj = j + step * dc
                        if ni < 0 or ni >= H or nj < 0 or nj >= W:
                            valid = False
                            break
                        if grid[ni][nj] != target[step]:
                            valid = False
                            break
                        positions.append((ni, nj))
                    if valid:
                        for (r, c) in positions:
                            print(f"{r+1} {c+1}")
                        return

if __name__ == "__main__":
    main()