def calculate_black_area(A, B, C, D):
    # Calculate the area of the rectangle
    total_area = (C - A) * (D - B)

    # Calculate the number of black and white squares
    black_squares = 0
    for x in range(A, C):
        for y in range(B, D):
            # Check if the square is black
            if (x + y) % 2 == (x % 2) ^ (y % 2):
                black_squares += 1

    # Calculate the area of the black region
    black_area = black_squares

    # Return twice the area of the black region
    return 2 * black_area


# Read the inputs from stdin
A, B, C, D = map(int, input().split())

# Calculate and print the result
print(calculate_black_area(A, B, C, D))