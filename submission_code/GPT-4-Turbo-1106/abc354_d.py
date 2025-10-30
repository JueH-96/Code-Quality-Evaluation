def calculate_black_area(A, B, C, D):
    # Calculate the total area of the rectangle
    total_area = (C - A) * (D - B)
    
    # Calculate the number of vertical and horizontal lines
    vertical_lines = C - A
    horizontal_lines = (D - B) // 2
    
    # Calculate the number of diagonal lines within the rectangle
    diagonal_lines = min((C + D) // 2 - (A + B + 1) // 2 + 1, vertical_lines, horizontal_lines)
    
    # Calculate the area of black regions
    black_area = (total_area + diagonal_lines) // 2
    
    return black_area * 2

# Read the inputs from stdin
A, B, C, D = map(int, input().split())

# Solve the problem and write the answer to stdout
print(calculate_black_area(A, B, C, D))