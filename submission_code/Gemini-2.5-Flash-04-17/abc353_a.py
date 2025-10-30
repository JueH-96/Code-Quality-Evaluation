# Read the number of buildings
N = int(input())

# Read the heights of the buildings
heights = list(map(int, input().split()))

# Get the height of the first building
if N > 0:
    first_height = heights[0]
else:
    # This case should not happen based on constraints (N >= 1),
    # but handling it defensively.
    print(-1)
    exit()

# Initialize a variable to store the index of the leftmost taller building
leftmost_taller_index = -1

# Iterate through the buildings starting from the second one (index 1)
for i in range(1, N):
    # Check if the current building is taller than the first one
    if heights[i] > first_height:
        # If it is, store its 1-based index and break the loop
        # because we are looking for the leftmost one
        leftmost_taller_index = i + 1
        break

# Print the result
print(leftmost_taller_index)