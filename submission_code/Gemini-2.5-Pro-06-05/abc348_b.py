import sys

def solve():
    """
    Reads N points from standard input. For each point, it finds the ID of the
    farthest point and prints it to standard output.
    """
    try:
        # Read the number of points.
        N = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # Handle cases with no or invalid input for N.
        return

    # Read and store the coordinates of all N points.
    # The list index `i` will correspond to the point with ID `i+1`.
    points = []
    for _ in range(N):
        line = sys.stdin.readline().split()
        if not line:
            # Reached end of input unexpectedly
            return
        x, y = map(int, line)
        points.append((x, y))

    # For each point `i`, find its farthest neighbor.
    for i in range(N):
        xi, yi = points[i]
        
        max_squared_dist = -1
        farthest_id = -1
        
        # Compare point `i` with every other point `j`.
        for j in range(N):
            # A point's distance to itself is 0, so we skip it.
            if i == j:
                continue
            
            xj, yj = points[j]
            
            # Calculate the squared Euclidean distance.
            # This avoids floating-point math and is sufficient for comparisons.
            squared_dist = (xi - xj)**2 + (yi - yj)**2
            
            # If we find a new farthest point, update our records.
            # The tie-breaking rule (smallest ID) is handled automatically
            # because we iterate through `j` in increasing order. The first
            # point found at a maximum distance will have the smallest ID.
            if squared_dist > max_squared_dist:
                max_squared_dist = squared_dist
                farthest_id = j + 1  # Point IDs are 1-based.
        
        # Print the ID of the farthest point.
        print(farthest_id)

solve()