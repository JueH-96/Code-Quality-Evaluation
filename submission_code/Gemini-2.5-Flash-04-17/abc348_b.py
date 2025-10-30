# YOUR CODE HERE
import sys

# Read N
N = int(sys.stdin.readline())

# Read points
# Store points as a list of tuples (x, y). Using 0-indexed list for convenience.
points = []
for _ in range(N):
    x, y = map(int, sys.stdin.readline().split())
    points.append((x, y))

# Process each point (from i=0 to N-1)
for i in range(N):
    px, py = points[i]

    # Initialize max squared distance found so far for point i
    # Initialize with a value smaller than any possible squared distance.
    # The minimum squared distance between distinct points is >= 1.
    max_dist_sq = -1

    # Initialize the 0-indexed ID of the farthest point found so far for point i.
    farthest_point_idx = -1

    # Compare point i with every other point j (from j=0 to N-1)
    for j in range(N):
        if i == j:
            continue  # A point is not farthest from itself, skip self-comparison.

        qx, qy = points[j]

        # Calculate squared Euclidean distance between point i and point j
        # Using squared distance avoids float square roots and preserves comparison result.
        dx = px - qx
        dy = py - qy
        dist_sq = dx*dx + dy*dy

        # Check if this point j is farther than the current farthest point,
        # or equally far but with a smaller ID.

        if dist_sq > max_dist_sq:
            # Found a new farthest point
            max_dist_sq = dist_sq
            farthest_point_idx = j # Update the 0-indexed ID of the farthest point

        elif dist_sq == max_dist_sq:
            # Found a point equally far as the current farthest.
            # Apply the tie-breaking rule: choose the one with the smallest 1-indexed ID.
            # This is equivalent to choosing the one with the smaller 0-indexed ID.
            # If the current point j has a smaller 0-indexed ID than the current farthest_point_idx, update.
            if j < farthest_point_idx:
                 farthest_point_idx = j

    # After checking all other points, farthest_point_idx holds the 0-indexed ID
    # of the farthest point from point i, applying the tie-breaking rule.
    # Print the 1-indexed ID (0-indexed ID + 1).
    print(farthest_point_idx + 1)