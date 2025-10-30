def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data:
        return
    R = float(data[0])
    R2 = R * R
    total = 0

    # We count squares by using the fact that a square with center (i,j) is entirely inside the circle
    # if and only if its farthest vertex from the origin has distance at most R.
    # In any quadrant, the farthest vertex for a square whose center has coordinates (|i|,|j|)
    # is (|i|+0.5, |j|+0.5). Hence the condition becomes:
    #      (|i| + 0.5)^2 + (|j| + 0.5)^2 <= R^2.
    # We iterate over nonnegative integer x = |i| and find the maximum y = |j| such that
    #      (x+0.5)^2 + (y+0.5)^2 <= R^2.
    # Solve for y gives:
    #      y <= sqrt(R^2 - (x+0.5)^2) - 0.5.
    # Let y_max be the maximum integer y satisfying that.
    # Then, by symmetry:
    #  • If x == 0: the square for y = 0 is unique and for each y>=1 we have two squares (one for positive j and one for negative).
    #  • If x > 0: for y == 0 there are 2 squares (x positive and negative) and for each y>=1 there are 4 squares.
    
    x = 0
    while (x + 0.5) ** 2 <= R2:
        # Calculate the largest nonnegative integer y such that (x+0.5)^2 + (y+0.5)^2 <= R^2.
        max_y_float = math.sqrt(R2 - (x + 0.5) ** 2) - 0.5
        y_max = int(math.floor(max_y_float))
        if y_max < 0:
            break
        if x == 0:
            # y = 0 contributes 1 square;
            # each y from 1 to y_max contributes 2 squares (mirrored about the x-axis)
            total += 1 + y_max * 2
        else:
            # for x > 0: y = 0 contributes 2 squares (mirrored about the y-axis),
            # and each y>=1 contributes 4 squares (mirrored in both axes)
            total += 2 + y_max * 4
        x += 1

    sys.stdout.write(str(total))
    
if __name__ == '__main__':
    main()