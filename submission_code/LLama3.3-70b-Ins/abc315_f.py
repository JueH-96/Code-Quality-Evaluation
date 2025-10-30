import sys
import math

def calculate_distance(x1, y1, x2, y2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_penalty(skipped_checkpoints):
    """Calculate the penalty for skipping checkpoints."""
    if skipped_checkpoints > 0:
        return 2 ** (skipped_checkpoints - 1)
    else:
        return 0

def main():
    # Read the number of checkpoints
    N = int(sys.stdin.readline())

    # Read the coordinates of each checkpoint
    checkpoints = []
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().split())
        checkpoints.append((x, y))

    # Initialize the minimum distance
    min_distance = float('inf')

    # Iterate over all possible subsets of checkpoints
    for mask in range(1 << (N - 2)):
        # Initialize the current distance and the number of skipped checkpoints
        current_distance = 0
        skipped_checkpoints = 0

        # Initialize the previous checkpoint
        prev_checkpoint = checkpoints[0]

        # Iterate over the checkpoints
        for i in range(1, N):
            # Check if the current checkpoint is included in the subset
            if i == N - 1 or (mask & (1 << (i - 1))):
                # Update the current distance
                current_distance += calculate_distance(prev_checkpoint[0], prev_checkpoint[1], checkpoints[i][0], checkpoints[i][1])
                # Update the previous checkpoint
                prev_checkpoint = checkpoints[i]
            else:
                # Increment the number of skipped checkpoints
                skipped_checkpoints += 1

        # Update the minimum distance
        min_distance = min(min_distance, current_distance + calculate_penalty(skipped_checkpoints))

    # Print the minimum distance
    print(min_distance)

if __name__ == "__main__":
    main()