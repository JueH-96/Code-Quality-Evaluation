def main():
    import sys

    # Read points A, B, C
    points = [tuple(map(int, sys.stdin.readline().split())) for _ in range(3)]
    A, B, C = points

    # Function to calculate squared distance between two points
    def squared_distance(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    # Calculate squared lengths of the sides
    ab = squared_distance(A, B)
    bc = squared_distance(B, C)
    ca = squared_distance(C, A)

    # Sort the squared lengths
    sides = sorted([ab, bc, ca])

    # Check for right triangle using Pythagorean theorem
    if sides[0] + sides[1] == sides[2]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()