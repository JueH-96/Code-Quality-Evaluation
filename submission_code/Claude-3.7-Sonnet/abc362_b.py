def is_right_triangle(A, B, C):
    # Calculate squared distances between each pair of points
    AB_squared = (B[0] - A[0])**2 + (B[1] - A[1])**2
    BC_squared = (C[0] - B[0])**2 + (C[1] - B[1])**2
    AC_squared = (C[0] - A[0])**2 + (C[1] - A[1])**2
    
    # Check if any angle is 90 degrees using the Pythagorean theorem
    # If angle at A is 90°: AB² + AC² = BC²
    # If angle at B is 90°: AB² + BC² = AC²
    # If angle at C is 90°: AC² + BC² = AB²
    if AB_squared + BC_squared == AC_squared or \
       AB_squared + AC_squared == BC_squared or \
       BC_squared + AC_squared == AB_squared:
        return "Yes"
    else:
        return "No"

# Read input
x_A, y_A = map(int, input().split())
x_B, y_B = map(int, input().split())
x_C, y_C = map(int, input().split())

# Create point tuples
A = (x_A, y_A)
B = (x_B, y_B)
C = (x_C, y_C)

# Check if it's a right triangle and print result
print(is_right_triangle(A, B, C))