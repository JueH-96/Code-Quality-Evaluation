# Read coordinates for point A
x_A, y_A = map(int, input().split())
# Read coordinates for point B
x_B, y_B = map(int, input().split())
# Read coordinates for point C
x_C, y_C = map(int, input().split())

# Define a helper function to calculate the squared distance between two points
def squared_distance(x1, y1, x2, y2):
    """Calculates the squared Euclidean distance between (x1, y1) and (x2, y2)."""
    delta_x = x1 - x2
    delta_y = y1 - y2
    return delta_x**2 + delta_y**2

# Calculate the squared lengths of the three sides of the triangle
# sAB_sq is the squared length of side AB
sAB_sq = squared_distance(x_A, y_A, x_B, y_B)
# sBC_sq is the squared length of side BC
sBC_sq = squared_distance(x_B, y_B, x_C, y_C)
# sCA_sq is the squared length of side CA
sCA_sq = squared_distance(x_C, y_C, x_A, y_A)

# Store the squared lengths in a list
list_squared_lengths = [sAB_sq, sBC_sq, sCA_sq]

# Sort the list in non-decreasing order
# This makes list_squared_lengths[0] and list_squared_lengths[1] the squares
# of the two shorter sides, and list_squared_lengths[2] the square of the longest side.
list_squared_lengths.sort()

# Check if the Pythagorean theorem holds: a^2 + b^2 = c^2
# The problem guarantees that the points are not collinear, meaning all side lengths
# are positive. Thus, list_squared_lengths[0] > 0.
if list_squared_lengths[0] + list_squared_lengths[1] == list_squared_lengths[2]:
    print("Yes")
else:
    print("No")