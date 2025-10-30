import sys

def solve():
    # Read the number of buildings
    n = int(sys.stdin.readline())

    # Read the heights of the buildings
    h = list(map(int, sys.stdin.readline().split()))

    # Get the height of the first building
    first_building_height = h[0]

    # Initialize the index of the taller building to -1 (default if none found)
    found_index = -1

    # Iterate through the buildings starting from the second one (index 1)
    # The loop variable 'i' represents the 0-based index.
    for i in range(1, n):
        # Check if the current building's height is greater than the first one
        if h[i] > first_building_height:
            # If found, store the 1-based position (i + 1)
            found_index = i + 1
            # Since we need the leftmost one, break the loop immediately
            break

    # Print the result
    print(found_index)

# Call the solve function to execute the logic
solve()