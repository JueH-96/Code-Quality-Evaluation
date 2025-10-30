# YOUR CODE HERE
import sys

def solve():
    # Read the number of sticks from standard input
    n = int(sys.stdin.readline())

    # Use a set to store the unique canonical representations of the sticks.
    # A set efficiently stores unique elements.
    unique_sticks = set()

    # Process each stick one by one
    for _ in range(n):
        # Read the string representing the current stick from standard input.
        # .strip() removes leading/trailing whitespace, including the newline character.
        s = sys.stdin.readline().strip()

        # Calculate the reverse of the string.
        # String slicing s[::-1] creates a reversed copy of the string s.
        rev_s = s[::-1]

        # Determine the canonical representation for the stick.
        # Two sticks are the same if their string representations are equal or one is the reverse of the other.
        # To handle this, we define a canonical form for each stick (s, rev_s) pair.
        # A convenient canonical form is the lexicographically smaller string between s and rev_s.
        # This ensures that a stick and its reverse will always map to the same canonical representation.
        canonical_s = min(s, rev_s)

        # Add the canonical representation to the set.
        # If the canonical representation is already in the set, add() does nothing.
        # This way, the set only stores distinct canonical forms encountered so far.
        unique_sticks.add(canonical_s)

    # The final count of different sticks is simply the number of unique canonical representations stored in the set.
    # Print the size of the set to standard output.
    print(len(unique_sticks))

# Call the solve function to execute the logic
solve()