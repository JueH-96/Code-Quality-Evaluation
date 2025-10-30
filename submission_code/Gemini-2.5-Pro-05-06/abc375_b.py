import math

# Helper function to calculate Euclidean distance between two points (x1, y1) and (x2, y2)
def calculate_euclidean_distance(x1, y1, x2, y2):
    # Calculate differences in coordinates
    # These differences (delta_x, delta_y) will be integers since x1,y1,x2,y2 are integers.
    delta_x = x1 - x2
    delta_y = y1 - y2
    
    # Use math.hypot for robust calculation of sqrt(delta_x^2 + delta_y^2).
    # math.hypot(dx, dy) computes the Euclidean norm. It works correctly with integer arguments.
    return math.hypot(delta_x, delta_y)

# Read the number of points Takahashi will visit
N = int(input())

# Initialize the total cost of the journey
total_travel_cost = 0.0

# Takahashi's current position on the 2D plane. He starts at the origin (0,0).
current_pos_x = 0
current_pos_y = 0

# Iterate N times, once for each point Takahashi must visit in order.
for _ in range(N):
    # Read the coordinates of the next point from input
    next_point_x, next_point_y = map(int, input().split())
    
    # Calculate the cost (distance) to move from his current position to the next point
    segment_cost = calculate_euclidean_distance(current_pos_x, current_pos_y, next_point_x, next_point_y)
    
    # Add this segment's cost to the cumulative total cost
    total_travel_cost += segment_cost
    
    # Update Takahashi's current position to the point he just visited
    current_pos_x = next_point_x
    current_pos_y = next_point_y

# After visiting all N points, Takahashi is at the location of the Nth point.
# Now, calculate the cost to return from this last visited point back to the origin (0,0).
cost_to_return_to_origin = calculate_euclidean_distance(current_pos_x, current_pos_y, 0, 0)

# Add this final segment's cost to the total cost
total_travel_cost += cost_to_return_to_origin

# Print the final total cost.
# The problem's sample outputs show results with high precision (18 decimal places).
# We format the float output to match this level of precision.
print(f"{total_travel_cost:.18f}")