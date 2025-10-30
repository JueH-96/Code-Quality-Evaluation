def main():
    import sys, math
    # Read input.
    data = sys.stdin.read().split()
    if not data:
        return
    R = int(data[0])
    R2 = R * R  # We'll compare with squared distances.

    # We use symmetry: The circle is centered at the origin.
    # In the first quadrant (i >= 0, j >= 0) the farthest vertex of the square with lower left corner (i, j)
    # is (i+0.5, j+0.5). A square is fully inside the circle if:
    #   (i + 0.5)^2 + (j + 0.5)^2 <= R^2.
    #
    # We iterate over i >= 0. For each such i, the condition becomes:
    #   (j + 0.5)^2 <= R^2 - (i + 0.5)^2.
    # Hence, j can range from 0 to floor(sqrt(R^2 - (i + 0.5)^2) - 0.5).
    #
    # We only need to loop until i such that (i + 0.5)^2 <= R^2, which means i <= R - 0.5.
    i_max = int(math.floor(R - 0.5))
    count_first_quadrant = 0
    for i in range(i_max + 1):
        # Compute the max allowed j for given i.
        remaining = R2 - (i + 0.5)**2
        if remaining < 0:
            continue
        j_max = math.floor(math.sqrt(remaining) - 0.5)
        if j_max < 0:
            continue
        # For this i, j can be 0,1,...,j_max. So add (j_max + 1)
        count_first_quadrant += (j_max + 1)

    # Since the circle is symmetric in all four quadrants,
    # the total number of squares is four times the count for the first quadrant.
    total_squares = 4 * count_first_quadrant

    sys.stdout.write(str(total_squares))

if __name__ == '__main__':
    main()