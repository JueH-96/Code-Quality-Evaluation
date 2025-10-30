def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    D = int(data[2])
    grid = [data[i + 3] for i in range(H)]
    
    # Collect all floor positions
    floor_positions = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floor_positions.append((i, j))
    
    n = len(floor_positions)
    
    # Function to calculate humidified cells by a humidifier at (x, y)
    def humidified_cells(x, y):
        cells = set()
        for dx in range(-D, D + 1):
            for dy in range(-D, D + 1):
                if abs(dx) + abs(dy) <= D:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '.':
                        cells.add((nx, ny))
        return cells
    
    # Try all pairs of floor positions to place humidifiers
    max_humidified = 0
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = floor_positions[i]
            x2, y2 = floor_positions[j]
            humidified_set = humidified_cells(x1, y1) | humidified_cells(x2, y2)
            max_humidified = max(max_humidified, len(humidified_set))
    
    print(max_humidified)

if __name__ == "__main__":
    main()