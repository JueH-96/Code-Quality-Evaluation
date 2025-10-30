# YOUR CODE HERE
import sys

# Read the two strings S and T from standard input, separated by a space.
# input() reads the entire line from stdin.
# .split() splits the line by whitespace into a list of strings.
# We expect exactly two strings, which are assigned to S and T.
S, T = input().split()

# Check if the first string S is exactly equal to "AtCoder"
# AND the second string T is exactly equal to "Land".
# The comparison `==` in Python is case-sensitive.
if S == "AtCoder" and T == "Land":
    # If both conditions are true, print "Yes" to standard output.
    print("Yes")
else:
    # Otherwise (if at least one condition is false), print "No" to standard output.
    print("No")