# YOUR CODE HERE
def is_right_triangle(x1, y1, x2, y2, x3, y3):
    # Calculate the squared lengths of the sides
    a_squared = (x2 - x1)**2 + (y2 - y1)**2
    b_squared = (x3 - x2)**2 + (y3 - y2)**2
    c_squared = (x1 - x3)**2 + (y1 - y3)**2
    
    # Check if the sum of any two squared lengths equals the third
    if a_squared + b_squared == c_squared:
        return True
    if a_squared + c_squared == b_squared:
        return True
    if b_squared + c_squared == a_squared:
        return True
    return False

# Read input
x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

# Determine if it's a right triangle
if is_right_triangle(x_A, y_A, x_B, y_B, x_C, y_C):
    print("Yes")
else:
    print("No")