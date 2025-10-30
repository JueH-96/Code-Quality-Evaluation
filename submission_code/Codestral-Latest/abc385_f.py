import sys
from math import sqrt

input = sys.stdin.read
data = input().split()

n = int(data[0])
buildings = [(int(data[i]), int(data[i + 1])) for i in range(1, 2 * n + 1, 2)]

def can_see_all(height):
    for x, h in buildings:
        if height * (x - 1) < h:
            return False
    return True

def binary_search():
    left, right = 0.0, 1e9
    while right - left > 1e-9:
        mid = (left + right) / 2
        if can_see_all(mid):
            left = mid
        else:
            right = mid
    return left

if can_see_all(0):
    print(-1)
else:
    print(binary_search())