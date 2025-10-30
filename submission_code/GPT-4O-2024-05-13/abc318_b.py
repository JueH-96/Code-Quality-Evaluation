# YOUR CODE HERE
def calculate_area(N, sheets):
    # Create a 2D grid to mark the covered areas
    grid = [[0] * 101 for _ in range(101)]
    
    # Mark the covered areas on the grid
    for sheet in sheets:
        A, B, C, D = sheet
        for x in range(A, B):
            for y in range(C, D):
                grid[x][y] = 1
    
    # Calculate the total covered area
    total_area = sum(sum(row) for row in grid)
    
    return total_area

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
sheets = []
index = 1
for i in range(N):
    A = int(data[index])
    B = int(data[index + 1])
    C = int(data[index + 2])
    D = int(data[index + 3])
    sheets.append((A, B, C, D))
    index += 4

# Calculate and print the result
result = calculate_area(N, sheets)
print(result)