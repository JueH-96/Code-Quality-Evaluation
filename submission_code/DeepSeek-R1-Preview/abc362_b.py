# Read the three points
a_x, a_y = map(int, input().split())
b_x, b_y = map(int, input().split())
c_x, c_y = map(int, input().split())

# Calculate the dot products for each vertex
dot1 = (b_x - a_x) * (c_x - a_x) + (b_y - a_y) * (c_y - a_y)
dot2 = (a_x - b_x) * (c_x - b_x) + (a_y - b_y) * (c_y - b_y)
dot3 = (a_x - c_x) * (b_x - c_x) + (a_y - c_y) * (b_y - c_y)

# Check if any of the dot products is zero
if dot1 == 0 or dot2 == 0 or dot3 == 0:
    print("Yes")
else:
    print("No")