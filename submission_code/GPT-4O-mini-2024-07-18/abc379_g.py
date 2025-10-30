def count_valid_grids(H, W, S):
    MOD = 998244353
    
    # Directions for adjacent cells (right, down)
    directions = [(0, 1), (1, 0)]
    
    # Function to check if the current grid is valid
    def is_valid(grid):
        for i in range(H):
            for j in range(W):
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if ni < H and nj < W and grid[i][j] == grid[ni][nj]:
                        return False
        return True
    
    # Function to generate all possible grids
    def generate_grids(grid, idx):
        if idx == len(positions):
            if is_valid(grid):
                return 1
            return 0
        
        x, y = positions[idx]
        total_count = 0
        
        for num in range(1, 4):  # Try replacing '?' with 1, 2, or 3
            grid[x][y] = str(num)
            total_count += generate_grids(grid, idx + 1)
            total_count %= MOD
        
        return total_count
    
    # Prepare the grid and find positions of '?'
    grid = [list(row) for row in S]
    positions = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == '?']
    
    # Count valid grids
    result = generate_grids(grid, 0)
    
    print(result)

# Reading input
import sys
input = sys.stdin.read
data = input().splitlines()

H, W = map(int, data[0].split())
S = data[1:H+1]

count_valid_grids(H, W, S)