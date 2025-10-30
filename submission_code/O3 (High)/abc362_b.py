import sys

def squared_distance(x1, y1, x2, y2):
    """Return the squared Euclidean distance between two points."""
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def main() -> None:
    data = list(map(int, sys.stdin.read().strip().split()))
    if len(data) != 6:   # Guard against wrong input length
        return

    xA, yA, xB, yB, xC, yC = data

    # Squared lengths of the three sides
    ab = squared_distance(xA, yA, xB, yB)
    bc = squared_distance(xB, yB, xC, yC)
    ca = squared_distance(xC, yC, xA, yA)

    # Sort so that the largest squared length is last
    sides = sorted([ab, bc, ca])

    # Pythagorean theorem in squared form
    if sides[0] + sides[1] == sides[2]:
        print("Yes")
    else:
        print("No")


if __name__ == "__main__":
    main()