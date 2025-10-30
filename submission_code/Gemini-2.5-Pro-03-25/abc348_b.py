# YOUR CODE HERE
import sys

# Function to calculate squared Euclidean distance
# Using squared distance avoids floating point issues with sqrt() and is computationally faster.
# Since sqrt is monotonically increasing for non-negative values, maximizing distance
# is equivalent to maximizing squared distance.
def dist_sq(p1_x, p1_y, p2_x, p2_y):
    """Calculates the squared Euclidean distance between two points (x1, y1) and (x2, y2)."""
    dx = p1_x - p2_x
    dy = p1_y - p2_y
    return dx*dx + dy*dy

# Main function to solve the problem
def solve():
    """Reads input, finds the farthest point for each point according to the rules, 
       and prints the ID of the farthest point for each."""
       
    # Read the number of points
    n = int(sys.stdin.readline())
    
    # List to store the coordinates of the points.
    # The point with ID k (1-based) will be stored at index k-1 (0-based).
    points = []
    
    # Read coordinates for each point and store them in the list
    for _ in range(n):
        points.append(list(map(int, sys.stdin.readline().split())))

    # Iterate through each point 'i' (from 0 to n-1, representing points 1 to N).
    # For each point 'i', we need to find the point 'j' that is farthest away.
    for i in range(n):
        max_d2 = -1  # Initialize the maximum squared distance found so far for point 'i' to -1.
                     # Any valid distance squared will be >= 0.
        farthest_id = -1 # Initialize the 1-based ID of the farthest point found so far.

        # Get the coordinates of the current reference point 'i'.
        x1, y1 = points[i] 

        # Iterate through all possible target points 'j' (from 0 to n-1).
        for j in range(n):
            # A point cannot be the farthest from itself, so skip the case where i == j.
            if i == j:
                continue

            # Get the coordinates of the potential farthest point 'j'.
            x2, y2 = points[j] 

            # Calculate the squared distance between point 'i' and point 'j'.
            d2 = dist_sq(x1, y1, x2, y2)

            # Get the 1-based ID of the potential farthest point 'j'.
            current_id = j + 1 

            # Check if the current point 'j' is farther than the farthest point found so far.
            if d2 > max_d2:
                # If point 'j' is strictly farther, update the maximum distance
                # and store the ID of point 'j' as the new farthest point.
                max_d2 = d2
                farthest_id = current_id
            elif d2 == max_d2:
                # If point 'j' is equally far as the current farthest point,
                # we apply the tie-breaking rule: choose the point with the smaller ID.
                # Check if the ID of the current point 'j' is smaller than the ID
                # of the farthest point found so far.
                # Note: farthest_id will be > 0 here because max_d2 is >= 0,
                # meaning at least one point j!=i has been processed unless N=1 (which is ruled out by N>=2).
                if current_id < farthest_id:
                    # If the current point 'j' has a smaller ID, update farthest_id.
                    # max_d2 remains the same.
                    farthest_id = current_id

        # After checking all other points 'j' for the current reference point 'i',
        # print the ID of the farthest point found.
        print(farthest_id)

# Execute the main function when the script is run
if __name__ == "__main__":
    solve()
# YOUR CODE HERE