# YOUR CODE HERE
import sys

def solve():
    """
    Reads input values, processes them according to the problem specification,
    and prints the result to standard output.
    """
    # Read the first line of input: N, L, R.
    # N, the length of the sequence, is not strictly needed for the implementation
    # since we can directly iterate over the list read in the next step.
    try:
        _n, l, r = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # Handle cases with empty or malformed input lines gracefully.
        return

    # Read the second line of input: the sequence of integers A.
    # Exit if the line is empty.
    line2 = sys.stdin.readline()
    if not line2:
        return
    a_list = list(map(int, line2.split()))

    # For each element `val` in `a_list`, we need to find the integer `x`
    # in the range [l, r] that is closest to `val`.
    # This is equivalent to clamping `val` to the range [l, r].
    # The clamping logic can be implemented concisely as: min(r, max(l, val)).
    # We use a list comprehension to apply this to all elements of `a_list`.
    results = [min(r, max(l, val)) for val in a_list]

    # Print the computed values separated by spaces.
    # The `*` operator unpacks the `results` list, so each element is passed
    # as a separate argument to `print()`, which separates them with spaces
    # by default.
    print(*results)

# Execute the main logic.
solve()