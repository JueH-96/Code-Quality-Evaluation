def main():
    import sys

    # Read input points
    xA, yA = map(int, sys.stdin.readline().split())
    xB, yB = map(int, sys.stdin.readline().split())
    xC, yC = map(int, sys.stdin.readline().split())

    # Compute squared lengths of the sides
    def sqdist(x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return dx*dx + dy*dy

    dAB = sqdist(xA, yA, xB, yB)
    dBC = sqdist(xB, yB, xC, yC)
    dCA = sqdist(xC, yC, xA, yA)

    # Check Pythagorean condition for any permutation
    if dAB + dBC == dCA or dAB + dCA == dBC or dBC + dCA == dAB:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()