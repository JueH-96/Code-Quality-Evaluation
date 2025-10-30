import sys

def main():
    import sys
    import math

    def input():
        return sys.stdin.read()

    data = input().split()
    N = int(data[0])
    buildings = []
    for i in range(N):
        X = int(data[1 + 2*i])
        H = int(data[2 + 2*i])
        buildings.append( (X, H) )

    def can_see_all(h):
        max_slope = -math.inf
        for X, H in buildings:
            current_slope = (H + h) / X
            if current_slope <= max_slope + 1e-12:
                return False
            max_slope = current_slope
        return True

    # Check if we can see all buildings from h=0
    if can_see_all(0):
        print(-1)
        return

    # Binary search for minimal h where can_see_all(h) is True
    left = 0.0
    right = 1e14
    for _ in range(100):
        mid = (left + right) / 2
        if can_see_all(mid):
            right = mid
        else:
            left = mid

    # After binary search, right is the minimal h_min where can see all
    # Check if h_min is effectively 0
    if right < 1e-9:
        print("0.000000000000000000")
    else:
        print("{0:.18f}".format(right))

if __name__ == "__main__":
    main()