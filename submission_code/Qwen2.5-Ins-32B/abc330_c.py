import math

def find_min_difference(D):
    min_diff = float('inf')
    max_x = int(math.sqrt(D)) + 1
    for x in range(max_x):
        y_squared = D - x**2
        y = int(math.sqrt(y_squared))
        diff = abs(x**2 + y**2 - D)
        if diff < min_diff:
            min_diff = diff
    return min_diff

D = int(input())
print(find_min_difference(D))