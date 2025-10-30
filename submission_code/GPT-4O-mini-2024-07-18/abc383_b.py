def max_humidified_cells(H, W, D, grid):
    # Find all floor cells
    floor_cells = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == '.']
    
    max_humidified = 0
    num_floors = len(floor_cells)

    # Function to calculate humidified cells from two humidifier positions
    def humidified_count(h1, h2):
        humidified = set()
        
        # Add humidified cells from the first humidifier
        for i in range(H):
            for j in range(W):
                if abs(h1[0] - i) + abs(h1[1] - j) <= D:
                    if grid[i][j] == '.':
                        humidified.add((i, j))
        
        # Add humidified cells from the second humidifier
        for i in range(H):
            for j in range(W):
                if abs(h2[0] - i) + abs(h2[1] - j) <= D:
                    if grid[i][j] == '.':
                        humidified.add((i, j))
        
        return len(humidified)

    # Try all pairs of floor cells
    for i in range(num_floors):
        for j in range(i + 1, num_floors):
            h1 = floor_cells[i]
            h2 = floor_cells[j]
            humidified = humidified_count(h1, h2)
            max_humidified = max(max_humidified, humidified)

    return max_humidified

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

H, W, D = map(int, data[0].split())
grid = data[1:H+1]

# Calculate and print the result
result = max_humidified_cells(H, W, D, grid)
print(result)