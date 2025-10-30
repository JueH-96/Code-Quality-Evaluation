def main():
    import sys

    # Read input points
    xA, yA = map(int, sys.stdin.readline().split())
    xB, yB = map(int, sys.stdin.readline().split())
    xC, yC = map(int, sys.stdin.readline().split())

    # Function to compute squared distance between two points
    def dist_sq(x1, y1, x2, y2):
        return (x1 - x2)**2 + (y1 - y2)**2

    # Compute squared lengths of triangle sides
    AB_sq = dist_sq(xA, yA, xB, yB)
    BC_sq = dist_sq(xB, yB, xC, yC)
    CA_sq = dist_sq(xC, yC, xA, yA)

    # Check the Pythagorean relation for any permutation of the sides
    if (AB_sq + BC_sq == CA_sq) or (BC_sq + CA_sq == AB_sq) or (CA_sq + AB_sq == BC_sq):
        print("Yes")
    else:
        print("No")

# Call the main function
main()