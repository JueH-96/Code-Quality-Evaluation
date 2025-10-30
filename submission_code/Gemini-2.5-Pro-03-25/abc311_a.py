import sys

def solve():
    """
    Reads the input N and string S, then determines the minimum number of characters
    to check from the left until 'A', 'B', and 'C' have all appeared at least once.
    Prints the result to standard output.
    """
    # Read the length of the string (N).
    # Although N is given, we can also just use the length of the string S.
    # Reading it ensures we consume the input correctly.
    n = int(sys.stdin.readline())

    # Read the string S and remove any trailing whitespace (like newline).
    s = sys.stdin.readline().strip()

    # Use a set to efficiently track the unique characters encountered so far.
    # Sets provide fast addition and checking for existence.
    found_chars = set()

    # Iterate through the string S using an index `i` (0-based).
    # The number of characters checked up to index `i` is `i + 1`.
    for i in range(n):
        # Add the character at the current index `i` to the set.
        # If the character is already in the set, adding it again has no effect.
        found_chars.add(s[i])

        # Check if the size of the set has reached 3.
        # This means we have encountered 'A', 'B', and 'C' at least once.
        if len(found_chars) == 3:
            # If all three characters are found, the condition is met for the first time
            # at this point. The number of characters checked is `i + 1`.
            # Print the result (1-based index).
            print(i + 1)
            # Since we need the *first* time the condition is satisfied,
            # we can stop the iteration immediately after finding the answer.
            return

# Execute the solve function when the script is run.
solve()