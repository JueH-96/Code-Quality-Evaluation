# YOUR CODE HERE
import sys
input = sys.stdin.read

# Read input
data = input().strip().split()
N = int(data[0])
T = int(data[1])
A = int(data[2])

# Calculate remaining votes
remaining_votes = N - (T + A)

# Determine if the outcome is decided
if T > A + remaining_votes:
    print("Yes")
else:
    print("No")