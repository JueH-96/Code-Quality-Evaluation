import sys
import math

def main():
    """
    Solves the problem by calculating the total cost of a path.
    The path starts at the origin, visits N points in a given order,
    and returns to the origin. The cost is the Euclidean distance.
    """
    # Read the number of points, N, from standard input.
    try:
        N = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # Exit gracefully if input is empty or invalid.
        return

    # Initialize the total cost and the starting point (origin).
    total_cost = 0.0
    prev_x, prev_y = 0, 0

    # Iterate through the N waypoints provided in the input.
    for _ in range(N):
        # Read the coordinates of the next waypoint.
        curr_x, curr_y = map(int, sys.stdin.readline().split())

        # Calculate the cost (Euclidean distance) from the previous point to the current one.
        # math.hypot(dx, dy) is equivalent to math.sqrt(dx**2 + dy**2) but is
        # generally more accurate and numerically stable.
        cost_segment = math.hypot(prev_x - curr_x, prev_y - curr_y)
        
        # Add this segment's cost to the total.
        total_cost += cost_segment
        
        # Update the 'previous' point to the current one for the next iteration.
        prev_x, prev_y = curr_x, curr_y

    # After visiting all N points, the last visited point is (prev_x, prev_y).
    # Calculate the cost for the final segment: from the last point back to the origin (0,0).
    cost_to_origin = math.hypot(prev_x, prev_y)
    total_cost += cost_to_origin

    # Print the final total cost. The problem requires a high-precision answer,
    # so we format the float to a sufficient number of decimal places.
    print(f"{total_cost:.17f}")

if __name__ == "__main__":
    main()