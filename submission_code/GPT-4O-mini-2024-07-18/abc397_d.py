def find_pair(N):
    # We will iterate over possible values of y
    # and calculate x based on the equation x^3 - y^3 = N
    # which can be rearranged to x^3 = N + y^3
    # thus x = (N + y^3)^(1/3)
    
    y = 1
    while True:
        # Calculate y^3
        y_cubed = y ** 3
        
        # Calculate x^3
        x_cubed = N + y_cubed
        
        # Calculate x
        x = int(round(x_cubed ** (1/3)))
        
        # Check if x^3 - y^3 equals N
        if x > 0 and x ** 3 - y_cubed == N:
            return x, y
        
        # If x^3 is less than y^3, we can stop
        if x_cubed < y_cubed:
            break
        
        y += 1
    
    return -1

import sys

# Read input
N = int(sys.stdin.read().strip())

# Find the pair (x, y)
result = find_pair(N)

# Print the result
if result == -1:
    print(-1)
else:
    print(result[0], result[1])