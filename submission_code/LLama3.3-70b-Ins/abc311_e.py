import sys

def count_holeless_squares(H, W, N, holes):
    """
    Count the number of holeless squares in a grid.

    Args:
    H (int): The number of rows in the grid.
    W (int): The number of columns in the grid.
    N (int): The number of holed squares.
    holes (list): A list of tuples representing the coordinates of the holed squares.

    Returns:
    int: The number of holeless squares in the grid.
    """
    # Create a set of holed squares for efficient lookups
    hole_set = set(holes)

    # Initialize the count of holeless squares
    count = 0

    # Iterate over all possible top-left corners of a square
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            # Iterate over all possible sizes of a square
            for n in range(1, min(H - i + 1, W - j + 1) + 1):
                # Check if the square is holeless
                if all((i + k, j + l) not in hole_set for k in range(n) for l in range(n)):
                    count += 1

    return count

def main():
    # Read the input from stdin
    H, W, N = map(int, sys.stdin.readline().split())
    holes = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]

    # Count the number of holeless squares
    count = count_holeless_squares(H, W, N, holes)

    # Print the result to stdout
    print(count)

if __name__ == "__main__":
    main()