# YOUR CODE HERE
import sys

def solve():
    """
    Reads the input N and string S, checks for adjacent 'a' and 'b',
    and prints "Yes" or "No" accordingly.
    """
    # Read the integer N (length of the string)
    # While N is provided, it's not strictly necessary for this specific check,
    # as we can use the length of the string S directly or Python's 'in' operator.
    n = int(sys.stdin.readline())

    # Read the string S and remove any trailing newline characters
    s = sys.stdin.readline().strip()

    # Check if the substring "ab" or "ba" exists within the string S
    # The 'in' operator efficiently checks for substrings.
    if "ab" in s or "ba" in s:
        # If either "ab" or "ba" is found, print "Yes"
        print("Yes")
    else:
        # Otherwise, print "No"
        print("No")

# Call the solve function to execute the program logic
solve()