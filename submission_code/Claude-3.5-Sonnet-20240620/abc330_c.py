# YOUR CODE HERE
import math

D = int(input())

min_diff = float('inf')
sqrt_D = int(math.sqrt(D))

for x in range(sqrt_D + 2):
    y_squared = D - x*x
    if y_squared < 0:
        break
    y = int(math.sqrt(y_squared))
    
    for dy in [-1, 0, 1]:
        y_test = y + dy
        if y_test < 0:
            continue
        diff = abs(x*x + y_test*y_test - D)
        min_diff = min(min_diff, diff)
        
        if min_diff == 0:
            break
    
    if min_diff == 0:
        break

print(min_diff)