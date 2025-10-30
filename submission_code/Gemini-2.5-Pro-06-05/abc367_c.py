import sys
import itertools

def solve():
    """
    Reads problem specifications from stdin, finds all integer sequences
    satisfying the given conditions, and prints them to stdout in
    lexicographical order.
    """

    # Read N (length of sequence) and K (divisor for sum) from the first line.
    try:
        line = sys.stdin.readline()
        # If the line is empty, it signifies end of input.
        if not line.strip():
            return
        N, K = map(int, line.split())

        # Read the list of upper bounds R_1, R_2, ..., R_N.
        R = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        # This handles cases like malformed input or empty lines.
        return

    # Create a list of range objects. The element at 0-indexed position `i`
    # can be an integer from 1 to R[i], inclusive. The range function's
    # stop parameter is exclusive, so we use R[i] + 1.
    ranges = [range(1, r + 1) for r in R]

    # itertools.product(*ranges) computes the Cartesian product of the input ranges.
    # This is equivalent to nested for-loops iterating over each range, and it
    # generates all possible sequences of length N.
    # A crucial property of itertools.product is that it yields the resulting
    # tuples in lexicographical order, which is required by the problem.
    # The `*` operator unpacks the `ranges` list into separate arguments.
    # For example, if ranges = [r1, r2], this is product(r1, r2).
    for seq in itertools.product(*ranges):
        # For each generated sequence, check if it meets the sum condition.
        # The sum of all its elements must be a multiple of K.
        if sum(seq) % K == 0:
            # If the condition is met, print the sequence to standard output.
            # The `*seq` syntax unpacks the sequence (a tuple) into individual
            # arguments for the print function. By default, print separates
            # arguments with a space, matching the required output format.
            print(*seq)

# Execute the solution
solve()