# YOUR CODE HERE
import math

def count_squares_in_circle(R):
    count = 0
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            if (i + 0.5)**2 + (j + 0.5)**2 <= R**2 and \
               (i + 0.5)**2 + (j - 0.5)**2 <= R**2 and \
               (i - 0.5)**2 + (j + 0.5)**2 <= R**2 and \
               (i - 0.5)**2 + (j - 0.5)**2 <= R**2:
                count += 1
    return count

# Read input
import sys
input = sys.stdin.read
R = int(input().strip())

# Output the result
print(count_squares_in_circle(R))