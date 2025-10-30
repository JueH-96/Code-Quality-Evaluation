import math
import sys

# Function to calculate Euclidean distance between two points
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Read the number of checkpoints
N = int(input().strip())

# Read the coordinates of the checkpoints
checkpoints = [tuple(map(int, input().split())) for _ in range(N)]

# Initialize the minimum distance with a large number
min_distance = float('inf')

# Iterate over all possible combinations of skipping checkpoints
for skipped in range(1 << (N - 2)):
    # Calculate the penalty for the current combination
    penalty = 2 ** bin(skipped).count('1') if skipped > 0 else 0

    # Initialize the total distance for the current combination
    total_distance = 0

    # Start from the first checkpoint
    prev_x, prev_y = checkpoints[0]

    # Iterate over the checkpoints
    for i in range(1, N - 1):
        # If the checkpoint is not skipped, add the distance to the total
        if not skipped & (1 << (i - 1)):
            x, y = checkpoints[i]
            total_distance += euclidean_distance(prev_x, prev_y, x, y)
            prev_x, prev_y = x, y

    # Add the distance from the last visited checkpoint to the last checkpoint
    total_distance += euclidean_distance(prev_x, prev_y, *checkpoints[-1])

    # Add the penalty to the total distance
    total_distance += penalty

    # Update the minimum distance if the current total distance is smaller
    min_distance = min(min_distance, total_distance)

# Print the minimum distance
print(f"{min_distance:.10f}")