# YOUR CODE HERE
import sys

def solve():
    """
    Reads input N and S, and determines if 'a' and 'b' are adjacent.
    """
    try:
        # Read the length of the string (not strictly needed for this solution's logic,
        # but part of the specified input format).
        n = int(sys.stdin.readline())

        # Read the string S.
        s = sys.stdin.readline().strip()

        # Check if the substring "ab" or "ba" exists in S.
        # The 'in' operator is an efficient way to check for substrings.
        if "ab" in s or "ba" in s:
            print("Yes")
        else:
            print("No")

    except (IOError, ValueError) as e:
        # Handle potential errors with input reading, though not expected
        # under competitive programming constraints.
        # For a robust script, this is good practice.
        # sys.stderr.write(f"Error reading input: {e}
")
        pass

# Run the solution
solve()