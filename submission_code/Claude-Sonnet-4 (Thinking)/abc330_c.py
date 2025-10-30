import math

D = int(input())

min_diff = float('inf')

max_x = int(math.sqrt(D)) + 1000

for x in range(max_x + 1):
    remainder = D - x * x
    
    if remainder >= 0:
        # x^2 <= D, so we want y^2 closest to remainder
        y_exact = math.sqrt(remainder)
        y_floor = int(y_exact)
        y_ceil = y_floor + 1
        
        # Try both y_floor and y_ceil
        for y in [y_floor, y_ceil]:
            if y >= 0:
                diff = abs(x * x + y * y - D)
                min_diff = min(min_diff, diff)
    else:
        # x^2 > D, so the minimum is achieved when y = 0
        diff = x * x - D
        if diff >= min_diff:
            # It will only get worse for larger x
            break
        min_diff = min(min_diff, diff)

print(min_diff)