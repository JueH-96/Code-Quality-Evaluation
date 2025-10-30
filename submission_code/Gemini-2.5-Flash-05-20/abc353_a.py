# Read the number of buildings
N = int(input())

# Read the heights of the buildings into a list
H = list(map(int, input().split()))

# Get the height of the first building
first_building_height = H[0]

# Initialize the result to -1, which means no taller building has been found yet
result = -1

# Iterate through the buildings starting from the second one (index 1)
# The problem asks for the leftmost building taller than the first,
# so we need to find the first such building we encounter.
for i in range(1, N):
    # Check if the current building's height is greater than the first building's height
    if H[i] > first_building_height:
        # If it is, we found the leftmost such building.
        # Store its 1-based position (index + 1)
        result = i + 1
        # Since we need the leftmost, we can stop searching now
        break

# Print the result
print(result)