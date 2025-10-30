def remaining_walls(H, W, Q, queries):
    # Initialize a grid with all walls present
    walls = [[True] * W for _ in range(H)]
    
    for R_q, C_q in queries:
        R_q -= 1  # Convert to 0-based index
        C_q -= 1  # Convert to 0-based index
        
        if walls[R_q][C_q]:
            # If there's a wall at (R_q, C_q), destroy it
            walls[R_q][C_q] = False
        else:
            # Check in all four directions for the first wall to destroy
            # Up
            for i in range(R_q - 1, -1, -1):
                if walls[i][C_q]:
                    walls[i][C_q] = False
                    break
            
            # Down
            for i in range(R_q + 1, H):
                if walls[i][C_q]:
                    walls[i][C_q] = False
                    break
            
            # Left
            for j in range(C_q - 1, -1, -1):
                if walls[R_q][j]:
                    walls[R_q][j] = False
                    break
            
            # Right
            for j in range(C_q + 1, W):
                if walls[R_q][j]:
                    walls[R_q][j] = False
                    break
    
    # Count remaining walls
    remaining = sum(sum(row) for row in walls)
    return remaining

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

H, W, Q = map(int, data[0].split())
queries = [tuple(map(int, line.split())) for line in data[1:Q + 1]]

# Get the result
result = remaining_walls(H, W, Q, queries)

# Print the result
print(result)