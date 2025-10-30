# YOUR CODE HERE
import sys
input = sys.stdin.read

def can_see_all_buildings(h, buildings):
    max_slope = float('-inf')
    for x, H in buildings:
        slope = (H - h) / x
        if slope < max_slope:
            return False
        max_slope = slope
    return True

def find_max_height(buildings):
    left, right = 0, 10**9
    while right - left > 1e-9:
        mid = (left + right) / 2
        if can_see_all_buildings(mid, buildings):
            left = mid
        else:
            right = mid
    return left

def main():
    data = input().strip().split()
    N = int(data[0])
    buildings = []
    for i in range(N):
        X = int(data[2 * i + 1])
        H = int(data[2 * i + 2])
        buildings.append((X, H))
    
    if can_see_all_buildings(0, buildings):
        print("-1")
    else:
        max_height = find_max_height(buildings)
        print(f"{max_height:.18f}")

if __name__ == "__main__":
    main()