def main():
    # Read input points
    xA, yA = map(int, input().split())
    xB, yB = map(int, input().split())
    xC, yC = map(int, input().split())

    # Compute squared lengths of the sides
    dAB2 = (xA - xB) ** 2 + (yA - yB) ** 2
    dBC2 = (xB - xC) ** 2 + (yB - yC) ** 2
    dCA2 = (xC - xA) ** 2 + (yC - yA) ** 2

    # Sort the squared lengths
    sides = sorted([dAB2, dBC2, dCA2])

    # Check the Pythagorean theorem: a^2 + b^2 == c^2
    if sides[0] + sides[1] == sides[2]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()