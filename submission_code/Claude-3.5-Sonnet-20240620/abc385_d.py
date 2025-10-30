# YOUR CODE HERE
import sys

def is_between(a, b, c):
    return min(a, b) <= c <= max(a, b)

def count_houses_passed(sx, sy, ex, ey, houses):
    count = 0
    for hx, hy in houses:
        if (sx == ex and hx == sx and is_between(sy, ey, hy)) or \
           (sy == ey and hy == sy and is_between(sx, ex, hx)) or \
           (sx == hx and sy == hy) or (ex == hx and ey == hy):
            count += 1
    return count

# Read input
N, M, Sx, Sy = map(int, input().split())
houses = set(tuple(map(int, input().split())) for _ in range(N))

# Process Santa's movements
x, y = Sx, Sy
houses_passed = set()

for _ in range(M):
    D, C = input().split()
    C = int(C)
    
    prev_x, prev_y = x, y
    if D == 'U':
        y += C
    elif D == 'D':
        y -= C
    elif D == 'L':
        x -= C
    elif D == 'R':
        x += C
    
    houses_passed.update(house for house in houses if count_houses_passed(prev_x, prev_y, x, y, [house]) > 0)

# Output result
print(x, y, len(houses_passed))