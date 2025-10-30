import math

def is_square_inside_circle(i, j, R):
    # Check if all four corners of the square are inside the circle
    corners = [
        (i + 0.5, j + 0.5),
        (i + 0.5, j - 0.5),
        (i - 0.5, j + 0.5),
        (i - 0.5, j - 0.5)
    ]
    
    for x, y in corners:
        if x*x + y*y > R*R:
            return False
    return True

def count_squares(R):
    # We only need to check squares in the first quadrant due to symmetry
    # and then multiply by 4, plus handle the axis squares separately
    count = 0
    
    # Find the maximum possible i,j values we need to check
    # A square at position (i,j) has its furthest corner at distance sqrt((i±0.5)^2 + (j±0.5)^2)
    max_coord = math.ceil(R)
    
    # Count squares in first quadrant (positive i, positive j)
    for i in range(max_coord):
        for j in range(max_coord):
            if i == 0 and j == 0:
                continue  # Handle origin separately
            if is_square_inside_circle(i, j, R):
                if i == 0 or j == 0:
                    count += 2  # Squares on axes are counted twice
                else:
                    count += 4  # Other squares are counted four times (one in each quadrant)

    # Handle the square at origin separately
    if is_square_inside_circle(0, 0, R):
        count += 1

    return count

# Read input
R = int(input())
# Print output
print(count_squares(R))