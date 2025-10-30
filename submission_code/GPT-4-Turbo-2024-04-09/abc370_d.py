def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    Q = int(data[2])
    
    queries = []
    idx = 3
    for _ in range(Q):
        R = int(data[idx]) - 1
        C = int(data[idx + 1]) - 1
        queries.append((R, C))
        idx += 2
    
    # Grid to keep track of walls, initially all True (wall exists)
    grid = [[True] * W for _ in range(H)]
    
    # Process each query
    for R, C in queries:
        if grid[R][C]:
            # If there is a wall, destroy it
            grid[R][C] = False
        else:
            # No wall at (R, C), destroy the nearest walls in four directions
            # Up
            for i in range(R - 1, -1, -1):
                if grid[i][C]:
                    grid[i][C] = False
                    break
            
            # Down
            for i in range(R + 1, H):
                if grid[i][C]:
                    grid[i][C] = False
                    break
            
            # Left
            for j in range(C - 1, -1, -1):
                if grid[R][j]:
                    grid[R][j] = False
                    break
            
            # Right
            for j in range(C + 1, W):
                if grid[R][j]:
                    grid[R][j] = False
                    break
    
    # Count remaining walls
    remaining_walls = sum(sum(row) for row in grid)
    print(remaining_walls)

if __name__ == "__main__":
    main()