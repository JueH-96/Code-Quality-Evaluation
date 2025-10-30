import math

D = int(input())

min_diff = D

# Check x from 0 to slightly beyond sqrt(D)
for x in range(int(math.sqrt(D)) + 2):
    x_sq = x * x
    
    if x_sq > D:
        # Best y is 0
        min_diff = min(min_diff, x_sq - D)
    else:
        # Find y such that y^2 is closest to D - x_sq
        target = D - x_sq
        y = int(math.sqrt(target))
        
        # Check y and y+1
        for y_cand in [y, y + 1]:
            diff = abs(x_sq + y_cand * y_cand - D)
            min_diff = min(min_diff, diff)

print(min_diff)