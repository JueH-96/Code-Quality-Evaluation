# YOUR CODE HERE
import math

def min_abs_diff(D):
    min_diff = float('inf')
    for x in range(int(math.sqrt(D)) + 1):
        y_squared = D - x*x
        y = int(math.sqrt(y_squared))
        if y*y == y_squared:
            return 0
        min_diff = min(min_diff, abs(x*x + y*y - D))
    return min_diff

D = int(input())
print(min_abs_diff(D))