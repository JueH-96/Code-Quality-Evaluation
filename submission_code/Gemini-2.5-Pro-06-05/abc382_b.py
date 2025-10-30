import sys

def solve():
    """
    This function solves the cookie problem by simulating the process day by day.
    """
    # Read the number of boxes (N) and the number of days (D)
    try:
        n, d = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        return  # Exit if input is malformed or empty

    # Read the initial state of the boxes as a string
    s = sys.stdin.readline().strip()

    # Convert the immutable string to a mutable list of characters
    # to simulate the changing state of the boxes.
    boxes = list(s)

    # Simulate the process for D days
    for _ in range(d):
        # On each day, Takahashi eats the cookie in the rightmost box.
        # We find this box by searching from right to left (from index n-1 down to 0).
        for i in range(n - 1, -1, -1):
            # If a box contains a cookie ('@')
            if boxes[i] == '@':
                # Takahashi eats it, so the box becomes empty ('.')
                boxes[i] = '.'
                # He eats only one cookie per day, so we break the search
                # and move on to the next day.
                break

    # After D days, join the list of characters back into a single string
    # to represent the final state of the boxes.
    final_state = "".join(boxes)

    # Print the final state to standard output.
    print(final_state)

solve()