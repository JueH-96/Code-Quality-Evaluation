import sys

# Read input
N, M = map(int, input().split())
X = list(map(int, input().split()))
A = list(map(int, input().split()))

# Sort the cells by the number of stones in descending order
cells = sorted(zip(X, A), key=lambda x: x[1], reverse=True)

# Initialize the number of operations to 0
operations = 0

# Iterate through the cells
for i in range(M):
    # If the current cell has more than one stone
    if cells[i][1] > 1:
        # Move the extra stones to the next cell
        extra_stones = cells[i][1] - 1
        cells[i+1] = (cells[i+1][0], cells[i+1][1] + extra_stones)
        cells[i] = (cells[i][0], 1)
        operations += extra_stones

# Check if it's possible to reach the desired state
if sum(cell[1] for cell in cells) == N:
    print(operations)
else:
    print(-1)