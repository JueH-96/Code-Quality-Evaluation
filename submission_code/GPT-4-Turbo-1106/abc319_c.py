from itertools import permutations

# Function to check if Takahashi gets disappointed
def is_disappointed(order, grid):
    # Check rows
    for i in range(3):
        if grid[order[i*3]] == grid[order[i*3+1]] and grid[order[i*3]] != grid[order[i*3+2]]:
            return True
    # Check columns
    for i in range(3):
        if grid[order[i]] == grid[order[i+3]] and grid[order[i]] != grid[order[i+6]]:
            return True
    # Check diagonals
    if grid[order[0]] == grid[order[4]] and grid[order[0]] != grid[order[8]]:
        return True
    if grid[order[2]] == grid[order[4]] and grid[order[2]] != grid[order[6]]:
        return True
    return False

# Read input
grid = []
for _ in range(3):
    grid.extend(map(int, input().split()))

# Calculate the probability
total_permutations = 0
not_disappointed_count = 0
for order in permutations(range(9)):
    total_permutations += 1
    if not is_disappointed(order, grid):
        not_disappointed_count += 1

# Output the probability
probability = not_disappointed_count / total_permutations
print(f"{probability:.18f}")