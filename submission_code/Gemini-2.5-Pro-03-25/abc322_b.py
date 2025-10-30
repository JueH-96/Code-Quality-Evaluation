# YOUR CODE HERE
import sys

def solve():
    # Read N and M from the first line of input
    n, m = map(int, sys.stdin.readline().split())
    # Read string S from the second line of input and remove trailing newline
    s = sys.stdin.readline().strip()
    # Read string T from the third line of input and remove trailing newline
    t = sys.stdin.readline().strip()

    # Check if S is a prefix of T
    # This is true if the first N characters of T are equal to S
    is_prefix = (t[:n] == s)

    # Check if S is a suffix of T
    # This is true if the last N characters of T are equal to S
    # We slice T from index M-N to the end
    is_suffix = (t[m-n:] == s)

    # Determine the output based on the prefix and suffix checks
    if is_prefix and is_suffix:
        # S is both a prefix and a suffix
        print(0)
    elif is_prefix and not is_suffix:
        # S is a prefix but not a suffix
        print(1)
    elif not is_prefix and is_suffix:
        # S is a suffix but not a prefix
        print(2)
    else: # not is_prefix and not is_suffix
        # S is neither a prefix nor a suffix
        print(3)

# Call the solve function to execute the logic
solve()