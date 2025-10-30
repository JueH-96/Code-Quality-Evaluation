# YOUR CODE HERE
import sys

# Read the input string from standard input
S = sys.stdin.readline().strip()

# Check if the string S ends with the suffix "san"
if S.endswith("san"):
    # If it ends with "san", print "Yes"
    print("Yes")
else:
    # Otherwise, print "No"
    print("No")