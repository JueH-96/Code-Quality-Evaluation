# Read the number of buildings and their heights from stdin
N, *H = map(int, input().split())

# Find the height of the first building
first_height = H[0]

# Initialize the position of the leftmost taller building to -1
leftmost_taller = -1

# Iterate over the buildings from left to right
for i, height in enumerate(H):
    # Check if the current building is taller than the first one
    if height > first_height:
        # Update the position of the leftmost taller building
        leftmost_taller = i + 1
        # Break the loop as we've found the leftmost taller building
        break

# Print the position of the leftmost taller building
print(leftmost_taller)