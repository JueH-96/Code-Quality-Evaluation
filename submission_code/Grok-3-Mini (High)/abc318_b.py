import sys

# Read all input data
data = sys.stdin.read().split()
index = 0

# Read N
N = int(data[index])
index += 1

# Grid size for x and y indices (0 to 99)
grid_size = 100
covered = [[False for _ in range(grid_size)] for _ in range(grid_size)]  # covered[x][y]

# Process each rectangle
for _ in range(N):
    A = int(data[index])
    B = int(data[index + 1])
    C = int(data[index + 2])
    D = int(data[index + 3])
    index += 4
    
    # Mark the unit squares covered by this rectangle
    for x in range(A, B):
        for y in range(C, D):
            covered[x][y] = True

# Count the number of covered unit squares
total_covered = 0
for x in range(grid_size):
    for y in range(grid_size):
        if covered[x][y]:
            total_covered += 1

# Output the total covered area
print(total_covered)