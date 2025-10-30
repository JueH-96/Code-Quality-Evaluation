# YOUR CODE HERE
import sys

def can_see_all(h, buildings):
    for x, height in buildings:
        if h * x < height:
            return False
    return True

def binary_search(low, high, buildings):
    while high - low > 1e-10:
        mid = (low + high) / 2
        if can_see_all(mid, buildings):
            low = mid
        else:
            high = mid
    return low

# Read input
N = int(sys.stdin.readline())
buildings = []
for _ in range(N):
    x, h = map(int, sys.stdin.readline().split())
    buildings.append((x, h))

# Sort buildings by x-coordinate
buildings.sort()

# Check if all buildings are visible from height 0
if can_see_all(0, buildings):
    print(-1)
else:
    # Binary search for the maximum height
    max_height = binary_search(0, 1e10, buildings)
    print(f"{max_height:.15f}")