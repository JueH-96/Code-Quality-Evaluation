import math
import sys

# Function to calculate Euclidean distance between two points (x1, y1) and (x2, y2)
def distance(x1, y1, x2, y2):
    """Calculates the Euclidean distance between (x1, y1) and (x2, y2)."""
    # Calculate the difference in x and y coordinates
    # Cast to integers to ensure integer arithmetic for potentially large coordinates
    # although map already returns integers. Defensive cast.
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    dx = x1 - x2
    dy = y1 - y2
    
    # Calculate the squared distance. Using integer arithmetic here preserves precision 
    # for potentially large coordinate differences before the square root operation.
    # Python's integers have arbitrary precision, preventing overflow issues seen in
    # fixed-size integer types.
    dist_sq = dx*dx + dy*dy 
    
    # Calculate the square root to get the Euclidean distance.
    # math.sqrt takes a float or integer argument and returns a float.
    # It handles large integer inputs correctly by implicitly converting them to floats.
    return math.sqrt(dist_sq)

def solve():
    """Reads input, calculates the total journey cost, and prints the result."""
    
    # Read the number of points to visit from standard input
    # Using sys.stdin.readline for efficiency with potentially large input sizes
    n = int(sys.stdin.readline())

    # Initialize the total cost of the journey
    total_cost = 0.0
    
    # Initialize the starting position (Takahashi starts at the origin)
    current_x = 0
    current_y = 0

    # Iterate through each of the N points specified in the input
    for _ in range(n):
        # Read the coordinates (x, y) of the next point to visit
        # Using map combined with sys.stdin.readline().split() is a standard efficient way
        x, y = map(int, sys.stdin.readline().split())
        
        # Calculate the distance from the current position to the next point
        # using the helper function and add this distance to the total accumulated cost
        total_cost += distance(current_x, current_y, x, y)
        
        # Update the current position to the coordinates of the point just visited
        # This point becomes the starting point for the next segment of the journey
        current_x = x
        current_y = y

    # After visiting all N points, the journey requires returning to the origin.
    # Calculate the distance from the last visited point (current_x, current_y) 
    # back to the starting point (origin 0, 0) and add it to the total cost.
    total_cost += distance(current_x, current_y, 0, 0)

    # Print the final total cost to standard output. 
    # Format the output to 15 decimal places using an f-string. This ensures 
    # sufficient precision to meet the problem requirement of an absolute or 
    # relative error at most 10^-6. Printing more digits than strictly necessary
    # is generally safe for floating point comparisons in competitive programming.
    print(f"{total_cost:.15f}")

# Call the main function to solve the problem when the script is executed
solve()