def check_overlap(a, b, x, offset_a, offset_b):
    """Check if the overlap of a and b matches x."""
    h_x, w_x = len(x), len(x[0])
    h_a, w_a = len(a), len(a[0])
    h_b, w_b = len(b), len(b[0])
    offset_a_x, offset_a_y = offset_a
    offset_b_x, offset_b_y = offset_b

    # Create a grid to store the overlap
    grid = [['.' for _ in range(w_x)] for _ in range(h_x)]

    # Mark the positions of black squares from a
    for i in range(h_a):
        for j in range(w_a):
            if a[i][j] == '#':
                x_i, y_i = i + offset_a_x, j + offset_a_y
                if 0 <= x_i < h_x and 0 <= y_i < w_x:
                    grid[x_i][y_i] = '#'

    # Mark the positions of black squares from b
    for i in range(h_b):
        for j in range(w_b):
            if b[i][j] == '#':
                x_i, y_i = i + offset_b_x, j + offset_b_y
                if 0 <= x_i < h_x and 0 <= y_i < w_x:
                    grid[x_i][y_i] = '#'

    # Check if the grid matches x
    for i in range(h_x):
        for j in range(w_x):
            if grid[i][j] != x[i][j]:
                return False

    return True


def solve(a, b, x):
    """Solve the problem."""
    h_x, w_x = len(x), len(x[0])
    h_a, w_a = len(a), len(a[0])
    h_b, w_b = len(b), len(b[0])

    # Check all possible overlaps
    for offset_a_x in range(-h_a + 1, h_x):
        for offset_a_y in range(-w_a + 1, w_x):
            for offset_b_x in range(-h_b + 1, h_x):
                for offset_b_y in range(-w_b + 1, w_x):
                    if check_overlap(a, b, x, (offset_a_x, offset_a_y), (offset_b_x, offset_b_y)):
                        return True

    return False


def main():
    """Read the input and solve the problem."""
    # Read the input
    h_a, w_a = map(int, input().split())
    a = [input() for _ in range(h_a)]
    h_b, w_b = map(int, input().split())
    b = [input() for _ in range(h_b)]
    h_x, w_x = map(int, input().split())
    x = [input() for _ in range(h_x)]

    # Solve the problem
    if solve(a, b, x):
        print("Yes")
    else:
        print("No")


if __name__ == "__main__":
    main()