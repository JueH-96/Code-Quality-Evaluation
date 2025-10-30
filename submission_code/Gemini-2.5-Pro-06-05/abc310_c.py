import sys

def solve():
    """
    Reads stick representations from stdin, counts the number of
    distinct sticks, and prints the result to stdout.
    """
    # Read the total number of sticks, N.
    # Using sys.stdin.readline for faster I/O with large inputs.
    try:
        n_str = sys.stdin.readline()
        if not n_str:
            return  # Handle empty input
        n = int(n_str)
    except (ValueError, IndexError):
        return  # Handle malformed input for N

    # A set is used to store the canonical representations of the unique sticks.
    # Sets only store unique elements, making them perfect for this task.
    distinct_sticks = set()

    # Iterate N times to process each stick.
    for _ in range(n):
        # Read the string representation of a stick, removing the trailing newline.
        s = sys.stdin.readline().strip()

        # To handle the condition that a stick is the same as its reversal,
        # we use a "canonical representation". We define this as the
        # lexicographically smaller of the string and its reverse.
        # This ensures that both a string and its reversal map to the same
        # representative element. For example, both "abc" and "cba" will be
        # represented by "abc".
        canonical_form = min(s, s[::-1])

        # Add the canonical form to our set. If a stick of this type has been
        # seen before, its canonical form will already be in the set,
        # and the set will not be modified.
        distinct_sticks.add(canonical_form)

    # The number of different sticks is the number of unique canonical forms collected,
    # which corresponds to the size of the set.
    print(len(distinct_sticks))

if __name__ == "__main__":
    solve()