import sys
import math

def find_minimum_difference(D):
    min_diff = float('inf')
    
    # Iterate over possible values of x
    for x in range(int(math.sqrt(D)) + 1):
        x_squared = x * x
        # Calculate y^2 such that x^2 + y^2 is close to D
        y_squared = D - x_squared
        if y_squared < 0:
            y_squared = 0
        y = int(math.sqrt(y_squared))
        
        # Check the difference for y and y+1
        for y_candidate in [y, y + 1]:
            current_diff = abs(x_squared + y_candidate * y_candidate - D)
            if current_diff < min_diff:
                min_diff = current_diff
    
    return min_diff

# Read input
D = int(sys.stdin.read().strip())

# Find and print the minimum difference
print(find_minimum_difference(D))