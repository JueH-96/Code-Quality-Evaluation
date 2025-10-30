def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    Q = int(data[2])
    
    queries = []
    index = 3
    for _ in range(Q):
        R = int(data[index]) - 1
        C = int(data[index + 1]) - 1
        queries.append((R, C))
        index += 2
    
    # Initialize the grid with walls
    grid = [[True] * W for _ in range(H)]
    
    # Process each query
    for R_q, C_q in queries:
        if grid[R_q][C_q]:
            # Destroy the wall at (R_q, C_q)
            grid[R_q][C_q] = False
        else:
            # No wall at (R_q, C_q), destroy the first walls in each direction
            # Up
            for i in range(R_q - 1, -1, -1):
                if grid[i][C_q]:
                    grid[i][C_q] = False
                    break
            # Down
            for i in range(R_q + 1, H):
                if grid[i][C_q]:
                    grid[i][C_q] = False
                    break
            # Left
            for j in range(C_q - 1, -1, -1):
                if grid[R_q][j]:
                    grid[R_q][j] = False
                    break
            # Right
            for j in range(C_q + 1, W):
                if grid[R_q][j]:
                    grid[R_q][j] = False
                    break
    
    # Count remaining walls
    remaining_walls = sum(sum(row) for row in grid)
    print(remaining_walls)