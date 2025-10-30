import math

D = int(input())

min_diff = float('inf')

# We need to check x values up to around sqrt(D)
# But we should go a bit beyond in case x^2 > D gives a better result
max_x = int(math.sqrt(D)) + 2

for x in range(max_x + 1):
    x_squared = x * x
    
    # If x^2 > D, we still want to check with y = 0
    if x_squared > D:
        diff = x_squared - D
        min_diff = min(min_diff, diff)
    else:
        # x^2 <= D, so we want y such that x^2 + y^2 is close to D
        target = D - x_squared
        
        # Try y values around sqrt(target)
        y_approx = math.sqrt(target)
        
        # Check floor and ceil of y_approx
        for y in [int(y_approx), int(y_approx) + 1]:
            if y >= 0:
                sum_squares = x_squared + y * y
                diff = abs(sum_squares - D)
                min_diff = min(min_diff, diff)

print(min_diff)