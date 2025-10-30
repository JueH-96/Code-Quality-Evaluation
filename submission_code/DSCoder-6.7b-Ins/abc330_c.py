import sys
import math

def min_value(D):
    min_val = float('inf')
    for x in range(0, int(math.sqrt(D))+1):
        y = math.sqrt(D - x**2)
        if y == int(y):
            min_val = min(min_val, abs(x**2 + y**2 - D))
    return min_val

D = int(sys.stdin.readline().strip())
print(min_value(D))