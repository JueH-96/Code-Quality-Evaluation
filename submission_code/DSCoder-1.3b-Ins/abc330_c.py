import sys
import math

def find_min_distance(D):
    min_distance = float('inf')
    for x in range(int(math.sqrt(D)) + 1):
        for y in range(int(math.sqrt(D)) + 1):
            if abs(x**2 + y**2 - D) < min_distance:
                min_distance = abs(x**2 + y**2 - D)
    return min_distance

D = int(sys.stdin.readline().strip())
print(find_min_distance(D))