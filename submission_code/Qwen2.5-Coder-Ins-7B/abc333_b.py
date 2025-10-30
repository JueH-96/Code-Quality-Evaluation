# YOUR CODE HERE
import sys

# Read input from stdin
input_line = sys.stdin.read().strip()

# Split the input into two segments
segments = input_line.split()

# Extract the points from each segment
S1, S2 = segments[0]
T1, T2 = segments[1]

# Check if the segments are equal
if (S1 == T1 and S2 == T2) or (S1 == T2 and S2 == T1):
    print("Yes")
else:
    print("No")