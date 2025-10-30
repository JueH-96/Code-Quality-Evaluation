import math

def distance(x1, y1, x2, y2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def is_right_triangle(x_A, y_A, x_B, y_B, x_C, y_C):
    """Determine whether the triangle ABC is a right triangle."""
    # Calculate the lengths of the sides of the triangle
    AB = distance(x_A, y_A, x_B, y_B)
    BC = distance(x_B, y_B, x_C, y_C)
    AC = distance(x_A, y_A, x_C, y_C)

    # Check if the triangle is a right triangle using the Pythagorean theorem
    sides = [AB, BC, AC]
    sides.sort()
    return math.isclose(sides[0] ** 2 + sides[1] ** 2, sides[2] ** 2)

def main():
    # Read the input from stdin
    x_A, y_A = map(int, input().split())
    x_B, y_B = map(int, input().split())
    x_C, y_C = map(int, input().split())

    # Determine whether the triangle ABC is a right triangle
    if is_right_triangle(x_A, y_A, x_B, y_B, x_C, y_C):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()