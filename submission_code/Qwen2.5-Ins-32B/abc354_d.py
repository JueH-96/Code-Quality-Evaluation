# YOUR CODE HERE
def calculate_black_area(A, B, C, D):
    # Adjust the coordinates to make the calculations easier
    A, B, C, D = A - 0.5, B - 0.5, C - 0.5, D - 0.5

    # Calculate the number of full squares in the x and y directions
    full_squares_x = (C // 2) - (A // 2)
    full_squares_y = (D // 2) - (B // 2)

    # Calculate the area of full squares
    full_squares_area = full_squares_x * full_squares_y * 4

    # Calculate the partial areas on the edges
    partial_area_x = (C % 2) * (D // 2 - B // 2) * 2
    partial_area_y = (D % 2) * (C // 2 - A // 2) * 2

    # Calculate the corner areas
    corner_area = 0
    if A % 2 != 0 and B % 2 != 0:
        corner_area += 1
    if A % 2 != 0 and D % 2 != 0:
        corner_area += 1
    if C % 2 != 0 and B % 2 != 0:
        corner_area += 1
    if C % 2 != 0 and D % 2 != 0:
        corner_area += 1

    # Total area is the sum of full squares, partial areas, and corner areas
    total_area = full_squares_area + partial_area_x + partial_area_y + corner_area

    return total_area

# Read input
A, B, C, D = map(int, input().split())

# Calculate and print the result
print(calculate_black_area(A, B, C, D))