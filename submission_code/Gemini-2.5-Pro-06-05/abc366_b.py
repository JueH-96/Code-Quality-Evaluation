# YOUR CODE HERE
import sys
import itertools

def solve():
    """
    Reads horizontally written text and converts it to vertical writing,
    filling spaces with '*' and trimming trailing '*' from each vertical line.
    """
    try:
        # Read the number of strings. Constraints state N >= 1.
        N = int(sys.stdin.readline())

        # Read the N strings into a list.
        S = [sys.stdin.readline().strip() for _ in range(N)]
    except (ValueError, IndexError):
        # Gracefully handle empty or malformed input, though not expected by constraints.
        return

    # The problem describes a transformation where the last input string (S_N)
    # becomes the top row of a conceptual grid, S_{N-1} the second row, and so on,
    # with S_1 being the bottom row. Reversing the input list of strings achieves
    # this ordering, simplifying the logic.
    s_reversed = S[::-1]

    # We need to transpose this grid of characters. `itertools.zip_longest` is
    # the ideal tool for this. It iterates over several iterables (our strings)
    # in parallel, producing tuples with an item from each. When a shorter
    # iterable is exhausted, it uses the `fillvalue`.
    # The '*' operator unpacks the list `s_reversed` into separate arguments.
    # For example, `zip_longest("fghi", "de", "abc", fillvalue='*')` yields
    # `('f', 'd', 'a')`, `('g', 'e', 'b')`, `('h', '*', 'c')`, `('i', '*', '*')`.
    transposed_columns = itertools.zip_longest(*s_reversed, fillvalue='*')

    # Now, iterate through the generated columns (which are tuples of characters).
    for column_tuple in transposed_columns:
        # Join the characters in the tuple to form the raw column string.
        # For example, ('h', '*', 'c') becomes "h*c".
        raw_column_str = "".join(column_tuple)
        
        # The problem requires that output strings do not end with '*'.
        # The `rstrip('*')` method removes all trailing '*' characters.
        # For example, "i**".rstrip('*') results in "i", while
        # "h*c".rstrip('*') remains "h*c".
        final_column_str = raw_column_str.rstrip('*')
        
        # Print the processed string to standard output, followed by a newline.
        print(final_column_str)

# Execute the solution.
solve()