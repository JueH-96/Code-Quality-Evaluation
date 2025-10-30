import sys
import math

def can_see_all_buildings(buildings, h):
    n = len(buildings)
    max_slope = float('-inf')
    for i in range(n):
        x, height = buildings[i]
        slope = (height - h) / x
        if slope > max_slope:
            max_slope = slope
            max_idx = i

    for i in range(max_idx + 1, n):
        x, height = buildings[i]
        if (height - h) / x < max_slope:
            return False
    return True

def find_max_height(buildings):
    low, high = 0.0, 1e18
    eps = 1e-9

    while high - low > eps:
        mid = (low + high) / 2
        if can_see_all_buildings(buildings, mid):
            high = mid
        else:
            low = mid

    if can_see_all_buildings(buildings, 0):
        return -1
    else:
        return low

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    buildings = []
    index = 1
    for _ in range(N):
        x = int(data[index])
        h = int(data[index + 1])
        buildings.append((x, h))
        index += 2

    result = find_max_height(buildings)
    if result == -1:
        print(-1)
    else:
        print(f"{result:.20f}")

if __name__ == "__main__":
    main()