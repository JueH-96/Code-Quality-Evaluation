import math

def count_squares_inside_circle(R):
    count = 0
    for i in range(-int(R) - 1, int(R) + 2):
        # Determine the x-coordinate of the farthest corner based on the sign of i
        if i >= 0:
            corner_x = i + 0.5
        else:
            corner_x = i - 0.5
        
        # Calculate the maximum possible value for corner_y^2
        temp = R**2 - corner_x**2
        if temp < 0:
            continue
        
        max_abs_j_plus_sgn_j_half = math.sqrt(temp)
        
        # Calculate the range of valid j values
        j_max = math.floor(max_abs_j_plus_sgn_j_half - 0.5)
        j_min = math.ceil(-max_abs_j_plus_sgn_j_half + 0.5)
        
        # Add the count of valid j's
        count += max(0, j_max - j_min + 1)
    
    return count

R = int(input())
print(count_squares_inside_circle(R))