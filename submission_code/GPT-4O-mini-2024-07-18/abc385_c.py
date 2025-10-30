def max_illuminated_buildings(N, heights):
    from collections import defaultdict

    # Dictionary to store positions of each height
    height_positions = defaultdict(list)

    # Populate the dictionary with positions of each height
    for index, height in enumerate(heights):
        height_positions[height].append(index)

    max_count = 1  # At least one building can always be chosen

    # Check each height's positions
    for positions in height_positions.values():
        length = len(positions)
        if length < 2:
            continue  # If there's only one building of this height, skip

        # Check all pairs of positions to find common intervals
        for i in range(length):
            for j in range(i + 1, length):
                # Calculate the difference in positions
                diff = positions[j] - positions[i]
                # Calculate the number of buildings that can be selected with this interval
                count = 2  # We already have two buildings (positions[i] and positions[j])
                next_position = positions[j] + diff

                # Continue to find more buildings at equal intervals
                while next_position in positions:
                    count += 1
                    next_position += diff

                max_count = max(max_count, count)

    return max_count

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
heights = list(map(int, data[1:]))

# Get the result
result = max_illuminated_buildings(N, heights)

# Print the output
print(result)