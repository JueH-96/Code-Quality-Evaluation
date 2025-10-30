def count_squares_within_circle(R):
    count = 0
    # We need to check all integer pairs (i, j) such that the square centered at (i, j) is fully contained in the circle
    # The center of the square is at (i + 0.5, j + 0.5)
    # The four corners of the square are (i + 0.5, j + 0.5), (i + 0.5, j - 0.5), (i - 0.5, j + 0.5), (i - 0.5, j - 0.5)
    # We need to check that all these points are within the circle of radius R centered at (0, 0)
    
    # The distance from the origin to the center of the square (i + 0.5, j + 0.5) must be <= R
    # The distance from the origin to the corners must also be <= R
    # The maximum distance from the center of the square to its corners is sqrt(2) * 0.5 = sqrt(2)/2
    # Thus, we need to check that the distance from the origin to the center of the square is <= R - sqrt(2)/2
    
    # Since we are dealing with integer coordinates, we can iterate over possible (i, j) values
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            # Calculate the distance from the origin to the center of the square
            center_distance_squared = (i + 0.5) ** 2 + (j + 0.5) ** 2
            # Calculate the distance from the origin to the corners
            corner_distance_squared = (i + 0.5) ** 2 + (j + 0.5) ** 2
            if center_distance_squared <= R ** 2 and corner_distance_squared <= R ** 2:
                count += 1
    
    return count

import sys

# Read input
R = int(sys.stdin.read().strip())

# Calculate the number of squares completely contained in the circle
result = count_squares_within_circle(R)

# Print the result
print(result)