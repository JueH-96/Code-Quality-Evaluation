import math

# Read number of points
N = int(input())

# Initialize current position at origin
current_x, current_y = 0, 0
total_cost = 0.0

# Read and visit each point in order
for _ in range(N):
    x, y = map(int, input().split())
    
    # Calculate distance from current position to next point
    distance = math.sqrt((current_x - x)**2 + (current_y - y)**2)
    total_cost += distance
    
    # Update current position
    current_x, current_y = x, y

# Calculate distance from last point back to origin
distance_to_origin = math.sqrt(current_x**2 + current_y**2)
total_cost += distance_to_origin

# Print the result
print(total_cost)