# YOUR CODE HERE
import math
import sys

def solve():
    # Read N, the number of points to visit
    N = int(sys.stdin.readline())
    
    total_cost = 0.0
    
    # Initialize current position at the origin (0, 0)
    current_x = 0
    current_y = 0
    
    # Iterate through each of the N points
    # For each point (target_x, target_y), calculate the cost from the
    # current position to this point, add it to total_cost, and then update
    # current_x and current_y to this new point.
    for _ in range(N):
        # Read the coordinates of the next target point
        target_x, target_y = map(int, sys.stdin.readline().split())
        
        # Calculate the differences in coordinates (delta_x, delta_y)
        # from current position to the target position.
        dx = current_x - target_x
        dy = current_y - target_y
        
        # Calculate the Euclidean distance using the formula sqrt(dx^2 + dy^2)
        # and add it to the running total cost.
        # Using dx*dx + dy*dy for squaring is generally preferred over dx**2 + dy**2
        # for performance in competitive programming, though for floats it's less critical.
        total_cost += math.sqrt(dx*dx + dy*dy)
        
        # Update current position to the recently visited target point.
        current_x = target_x
        current_y = target_y
        
    # After visiting all N points, current_x and current_y hold the coordinates
    # of the N-th point (X_N, Y_N).
    # Now, calculate the cost to return from (X_N, Y_N) to the origin (0, 0).
    
    # Calculate the differences in coordinates for returning to origin.
    # Since the target is (0,0), dx is current_x - 0 = current_x, and dy is current_y - 0 = current_y.
    dx_to_origin = current_x
    dy_to_origin = current_y
    
    total_cost += math.sqrt(dx_to_origin*dx_to_origin + dy_to_origin*dy_to_origin)
            
    # Print the final total cost.
    # Python's default float printing is usually sufficient for competitive programming
    # requirements of precision unless specific formatting is requested.
    print(total_cost)

# Call the solve function to execute the program
if __name__ == '__main__':
    solve()