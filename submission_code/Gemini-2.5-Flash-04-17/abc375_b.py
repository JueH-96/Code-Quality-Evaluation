# YOUR CODE HERE
import math
import sys

# Read N
N = int(sys.stdin.readline())

# Initialize current position and total cost
current_x = 0
current_y = 0
total_cost = 0.0

# Iterate through the N points
for _ in range(N):
    # Read the next point's coordinates
    line = sys.stdin.readline().split()
    X = int(line[0])
    Y = int(line[1])

    # Calculate the distance from the current point to (X, Y) using math.hypot
    # dx = current_x - X
    # dy = current_y - Y
    # math.hypot(a, b) calculates sqrt(a*a + b*b) robustly.
    # The differences (current_x - X) and (current_y - Y) are integers.
    # math.hypot accepts float arguments, and integer arguments are automatically converted.
    # Since the differences are at most 2 * 10^9, which is less than 2^53,
    # these integers are exactly representable as 64-bit floats.
    distance = math.hypot(current_x - X, current_y - Y)

    # Add the distance to the total cost
    total_cost += distance

    # Update the current position
    current_x = X
    current_y = Y

# Calculate the distance from the last point (X_N, Y_N) back to the origin (0, 0)
# The current position is now (X_N, Y_N)
# distance_back_to_origin = math.hypot(current_x - 0, current_y - 0) is equivalent to math.hypot(current_x, current_y)
distance_back_to_origin = math.hypot(current_x, current_y)

# Add the final distance to the total cost
total_cost += distance_back_to_origin

# Print the total cost
# Print with enough precision to meet the requirement.
# The sample output suggests ~20 decimal places.
# Using f-string formatting with a specific precision.
print(f"{total_cost:.20f}")