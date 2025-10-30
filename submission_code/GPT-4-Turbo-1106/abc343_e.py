def find_cubes(V1, V2, V3):
    # The side length of the cubes
    side_length = 7
    cube_volume = side_length ** 3

    # Check if the volumes are possible
    if V3 > cube_volume or V2 > 2 * (cube_volume - side_length) or V1 > 3 * cube_volume - 3 * side_length - V2 - V3:
        return "No"

    # Start by placing the first cube at the origin
    a1, b1, c1 = 0, 0, 0

    # We need to find the overlap for V3, which is a cube of side length x
    # such that x^3 = V3. Since the cubes are integer-sided, x must be an integer.
    x = int(V3 ** (1/3))

    # If V3 is not a perfect cube, there's no solution
    if x ** 3 != V3:
        return "No"

    # Now we place the second cube such that it overlaps with the first one
    # in a cube of side length x. We can do this by shifting it by 7 - x
    # in any direction. Let's shift it along the x-axis.
    a2, b2, c2 = side_length - x, 0, 0

    # For V2, we need two faces to overlap. Since we already have an overlap
    # of x in the x-direction, we need an overlap of y in the y-direction
    # such that 2 * x * y = V2. We can solve for y.
    if x == 0:
        y = 0
    else:
        y = V2 // (2 * x)

    # If 2 * x * y != V2 or y is greater than 7, there's no solution
    if 2 * x * y != V2 or y > side_length:
        return "No"

    # Now we place the third cube such that it overlaps with the first one
    # in a face of side length y. We can do this by shifting it by 7 - y
    # in the y-direction.
    a3, b3, c3 = 0, side_length - y, 0

    # Now we need to check if the remaining volume is equal to V1
    remaining_volume = 3 * cube_volume - x ** 3 - 2 * x * y * side_length
    if remaining_volume != V1:
        return "No"

    return f"Yes
{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}"

# Read the inputs from stdin
V1, V2, V3 = map(int, input().split())

# Solve the problem and write the answer to stdout
print(find_cubes(V1, V2, V3))