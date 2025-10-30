# Read input
N = int(input())
heights = list(map(int, input().split()))

# Get the height of the first building
first_height = heights[0]

# Initialize position to -1
position = -1

# Iterate through the buildings starting from the second one (index 1)
for idx in range(1, len(heights)):
    if heights[idx] > first_height:
        position = idx + 1  # Convert to 1-based index
        break  # Stop after finding the first taller building

# Output the result
print(position)