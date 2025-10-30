def main():
    # Read the coordinates of the three points
    xA, yA = map(int, input().split())
    xB, yB = map(int, input().split())
    xC, yC = map(int, input().split())

    # Compute squared lengths of the sides AB, BC, and CA
    d2_AB = (xA - xB)**2 + (yA - yB)**2
    d2_BC = (xB - xC)**2 + (yB - yC)**2
    d2_CA = (xC - xA)**2 + (yC - yA)**2

    # Sort squared lengths so that the largest is last
    sides_sq = sorted([d2_AB, d2_BC, d2_CA])

    # Check the Pythagorean theorem: a^2 + b^2 == c^2
    if sides_sq[0] + sides_sq[1] == sides_sq[2]:
        print("Yes")
    else:
        print("No")

# Call main to execute
main()