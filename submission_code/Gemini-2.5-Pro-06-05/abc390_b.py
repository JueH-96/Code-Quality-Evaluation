import sys

def solve():
    """
    Reads a sequence of integers and determines if it is a geometric progression.
    """
    # Read the number of elements from standard input.
    try:
        n = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # Handle cases where input is empty or not an integer.
        return

    # Read the sequence of space-separated integers from standard input.
    try:
        a = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        # Handle cases where the line is empty or contains non-integer values.
        return

    # A sequence is a geometric progression if the ratio of consecutive terms is constant.
    # To avoid floating-point division and potential precision errors, we check the
    # equality of ratios using cross-multiplication:
    # A[i] / A[i-1] == A[i-1] / A[i-2] is equivalent to A[i] * A[i-2] == A[i-1] * A[i-1].
    # This must hold for i from 2 to n-1 (0-based index).
    #
    # The `all()` function with a generator expression is a concise and efficient
    # way to verify this. If n <= 2, range(2, n) is empty, and all([]) is True,
    # correctly identifying short sequences as geometric progressions.
    # Python's arbitrary-precision integers prevent overflow from the multiplication,
    # as products can reach up to (10^9)^2 = 10^18.
    if all(a[i] * a[i - 2] == a[i - 1] * a[i - 1] for i in range(2, n)):
        print("Yes")
    else:
        print("No")

solve()