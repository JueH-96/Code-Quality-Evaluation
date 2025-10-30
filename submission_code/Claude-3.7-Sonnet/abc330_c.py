import math

def min_difference(D):
    min_diff = float('inf')
    
    # We only need to check x values up to sqrt(D) + some small buffer
    for x in range(int(math.sqrt(D)) + 2):
        # If x^2 is already too large, break early
        if x**2 > D + min_diff:
            break
            
        target_y_squared = D - x**2
        if target_y_squared < 0:
            # If x^2 > D, best y is 0
            diff = x**2 - D
        else:
            # Find the closest integer y to sqrt(target_y_squared)
            # Check both floor and ceiling
            y_floor = int(math.sqrt(target_y_squared))
            diff_floor = abs(x**2 + y_floor**2 - D)
            
            y_ceil = y_floor + 1
            diff_ceil = abs(x**2 + y_ceil**2 - D)
            
            diff = min(diff_floor, diff_ceil)
        
        min_diff = min(min_diff, diff)
    
    return min_diff

D = int(input())
print(min_difference(D))