# YOUR CODE HERE
import math

D = int(input())

min_diff = float('inf')

# We only need to check up to the square root of D for x and y
max_val = int(math.sqrt(D)) + 1

for x in range(max_val):
    for y in range(max_val):
        current_diff = abs(x**2 + y**2 - D)
        if current_diff < min_diff:
            min_diff = current_diff

print(min_diff)