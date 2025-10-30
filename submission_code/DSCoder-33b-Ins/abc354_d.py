# YOUR CODE HERE

# Read the inputs
A, B, C, D = map(int, input().split())

# Calculate the area of the black-painted region inside the rectangle
area = (C - A) * (D - B)

# Print twice that area
print(2 * area)