# YOUR CODE HERE
import sys
import collections

def solve():
    """
    Reads a string from standard input, checks if it's a 1122 string based on the given conditions,
    and prints "Yes" or "No" to standard output.
    """
    # Read input string from stdin
    s = sys.stdin.readline().strip()
    n = len(s)

    # Condition 1: Length must be even
    # Check if the length of the string is odd. If so, it cannot be a 1122 string.
    if n % 2 != 0:
        print("No")
        return # Exit if condition 1 fails

    # Condition 2: Adjacent pairs must be equal
    # Iterate through the string with a step of 2, checking pairs of characters
    # at indices (0,1), (2,3), ..., (n-2, n-1).
    for i in range(0, n, 2):
        # If the characters at index i and i+1 are not the same
        if s[i] != s[i+1]:
            print("No")
            return # Exit if condition 2 fails

    # Condition 3: Each character must appear exactly twice
    # Use collections.Counter to efficiently count the frequency of each character in the string.
    counts = collections.Counter(s)
    # Iterate through the frequency counts of all characters present in the string.
    for char_count in counts.values():
        # If the count of any character is not exactly 2
        if char_count != 2:
            print("No")
            return # Exit if condition 3 fails

    # If all three conditions passed successfully
    print("Yes")

# Call the solve function to execute the logic when the script is run
solve()