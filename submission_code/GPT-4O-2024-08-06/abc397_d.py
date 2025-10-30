import sys
import math

def find_xy(N):
    # Iterate over possible values of y
    y = 1
    while True:
        y_cubed = y ** 3
        if y_cubed > N:
            break
        
        x_cubed = N + y_cubed
        x = round(x_cubed ** (1/3))
        
        # Check if x^3 is exactly x_cubed
        if x > y and x ** 3 == x_cubed:
            return x, y
        
        y += 1
    
    return -1

# Read input
N = int(sys.stdin.read().strip())

# Find the pair (x, y)
result = find_xy(N)

# Print the result
if result == -1:
    print(-1)
else:
    print(result[0], result[1])