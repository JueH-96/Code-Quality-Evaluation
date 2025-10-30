import math

D = int(input())

min_diff = float('inf')

# Check x values from 0 to sqrt(D)
max_x = int(math.sqrt(D)) + 1

for x in range(max_x + 1):
    x_squared = x * x
    
    if x_squared >= D:
        # If x² >= D, best y is 0
        diff = abs(x_squared - D)
        min_diff = min(min_diff, diff)
    else:
        # Find best y for this x
        remaining = D - x_squared
        y_exact = math.sqrt(remaining)
        
        # Check floor value
        y = int(y_exact)
        diff = abs(x_squared + y * y - D)
        min_diff = min(min_diff, diff)
        
        # Check ceiling value
        y += 1
        diff = abs(x_squared + y * y - D)
        min_diff = min(min_diff, diff)

print(min_diff)