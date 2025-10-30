# YOUR CODE HERE
import sys

# Read the two strings from standard input
s = sys.stdin.readline().strip()
t = sys.stdin.readline().strip()

# Check if the strings are identical
if s == t:
    # If they are identical, print 0
    print(0)
else:
    # If they are not identical, find the first differing position
    ans = 0 # Initialize the answer variable (will be updated below)
    len_s = len(s) # Length of string S
    len_t = len(t) # Length of string T
    min_len = min(len_s, len_t) # Minimum length of the two strings

    # Iterate through the characters up to the minimum length
    # We are looking for the smallest position i where the characters differ
    # or where one string ends while the other continues.
    found_diff = False
    for i in range(min_len):
        # Compare the characters at the current position (0-based index i)
        if s[i] != t[i]:
            # If characters differ, the position is i + 1 (1-based index)
            ans = i + 1
            found_diff = True
            # Break the loop as we found the first difference
            break

    # If the loop completed without finding a difference within the common prefix
    if not found_diff:
        # This means one string is a proper prefix of the other (since we already checked for equality).
        # The first difference occurs at the position immediately following the shorter string.
        # This corresponds to index min_len, which is position min_len + 1.
        ans = min_len + 1

    # Print the calculated first differing position
    print(ans)