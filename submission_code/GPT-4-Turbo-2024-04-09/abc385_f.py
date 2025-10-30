import sys
import math

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    buildings = []
    
    index = 1
    for _ in range(N):
        X = int(data[index])
        H = int(data[index + 1])
        buildings.append((X, H))
        index += 2
    
    # If we can see all buildings from height 0, return -1
    if all(buildings[i][0] > buildings[i][1] for i in range(N)):
        print(-1)
        return
    
    # Function to check if a building is visible from height h at x=0
    def is_visible(h, x, height):
        # Line from (0, h) to (x, height)
        # Slope of this line is (height - h) / x
        # We need to check if this line is above all other buildings at their respective x
        for bx, bh in buildings:
            if bx == x:
                continue
            # Height at bx from the line
            visible_height = h + (height - h) * (bx / x)
            if visible_height <= bh:
                return False
        return True
    
    # Binary search for the maximum height h from which not all buildings are visible
    low = 0.0
    high = 1e9 + 1.0  # A bit more than the maximum possible height of any building
    
    while high - low > 1e-10:  # Precision requirement
        mid = (low + high) / 2
        if all(is_visible(mid, x, h) for x, h in buildings):
            low = mid
        else:
            high = mid
    
    print(f"{low:.18f}")

if __name__ == "__main__":
    main()