H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Directions to check: all possible (dx, dy) except (0, 0)
directions = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0), (1, 1)]

for i in range(H):
    for j in range(W):
        if grid[i][j] == 's':
            for dx, dy in directions:
                # Calculate positions for A2 to A5
                x2, y2 = i + dx, j + dy
                x3, y3 = i + 2*dx, j + 2*dy
                x4, y4 = i + 3*dx, j + 3*dy
                x5, y5 = i + 4*dx, j + 4*dy
                
                # Check if all positions are within grid bounds
                if (x2 < 0 or x2 >= H or y2 < 0 or y2 >= W or
                    x3 < 0 or x3 >= H or y3 < 0 or y3 >= W or
                    x4 < 0 or x4 >= H or y4 < 0 or y4 >= W or
                    x5 < 0 or x5 >= H or y5 < 0 or y5 >= W):
                    continue
                
                # Check if the letters match the required sequence
                if (grid[x2][y2] == 'n' and
                    grid[x3][y3] == 'u' and
                    grid[x4][y4] == 'k' and
                    grid[x5][y5] == 'e'):
                    
                    # Convert to 1-based indices for output
                    points = [
                        (i + 1, j + 1),
                        (x2 + 1, y2 + 1),
                        (x3 + 1, y3 + 1),
                        (x4 + 1, y4 + 1),
                        (x5 + 1, y5 + 1)
                    ]
                    
                    for p in points:
                        print(p[0], p[1])
                    exit()