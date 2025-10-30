# YOUR CODE HERE
A, B, C, D = map(int, input().split())

# Calculate the area of the black-painted region inside the rectangle
black_area = (C - A) * (D - B)

# Print twice the area
print(2 * black_area)