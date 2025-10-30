# YOUR CODE HERE
import sys

def solve():
    """
    Reads three points, calculates the squared lengths of the triangle's sides,
    and checks if they satisfy the Pythagorean theorem to determine if it's a
    right triangle.
    """
    # Read the coordinates of the three points A, B, and C.
    try:
        xa, ya = map(int, sys.stdin.readline().split())
        xb, yb = map(int, sys.stdin.readline().split())
        xc, yc = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # Handle potential empty lines or invalid input, though problem constraints
        # suggest this won't happen.
        return

    # Calculate the squared lengths of the three sides of the triangle.
    # Using squared lengths avoids floating-point arithmetic and potential precision errors.
    # The distance formula is sqrt((x2-x1)^2 + (y2-y1)^2).
    # The squared distance is (x2-x1)^2 + (y2-y1)^2.
    len_ab_sq = (xb - xa)**2 + (yb - ya)**2
    len_bc_sq = (xc - xb)**2 + (yc - yb)**2
    len_ac_sq = (xc - xa)**2 + (yc - ya)**2

    # Store the squared lengths in a list.
    sides_sq = [len_ab_sq, len_bc_sq, len_ac_sq]

    # Sort the list to easily identify the two shorter sides and the longest side.
    sides_sq.sort()

    # According to the Pythagorean theorem, a triangle is a right triangle if the
    # sum of the squares of the two shorter sides equals the square of the longest side.
    # After sorting, sides_sq[0] and sides_sq[1] are the squares of the two shorter sides,
    # and sides_sq[2] is the square of the longest side (the potential hypotenuse).
    # The problem statement guarantees the points are not collinear, so all side lengths are positive.
    if sides_sq[0] > 0 and sides_sq[0] + sides_sq[1] == sides_sq[2]:
        print("Yes")
    else:
        print("No")

solve()