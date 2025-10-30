# YOUR CODE HERE
import sys

def solve():
    """
    Reads N, S, T from stdin and prints the Hamming distance between S and T.
    """
    # Read the integer N, the length of the strings.
    # While not strictly needed for the zip-based solution, we must read it
    # to consume the input line as per the problem format.
    try:
        n_str = sys.stdin.readline()
        if not n_str: return # Handle empty input case
        n = int(n_str)

        # Read the two strings S and T.
        s = sys.stdin.readline().strip()
        t = sys.stdin.readline().strip()
    except (IOError, ValueError):
        # Handle potential errors during input reading, though not expected
        # under problem constraints.
        return

    # Initialize a counter for the Hamming distance.
    distance = 0

    # The most direct way: iterate with an index from 0 to N-1.
    # for i in range(n):
    #     if s[i] != t[i]:
    #         distance += 1

    # A more Pythonic way using zip and a generator expression.
    # zip(s, t) creates pairs of characters: (s[0], t[0]), (s[1], t[1]), ...
    # (char_s != char_t) evaluates to a boolean (True/False).
    # sum() treats True as 1 and False as 0.
    distance = sum(char_s != char_t for char_s, char_t in zip(s, t))

    # Print the final calculated distance to standard output.
    print(distance)

solve()