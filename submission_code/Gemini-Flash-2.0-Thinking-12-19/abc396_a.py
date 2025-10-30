# YOUR CODE HERE
import sys

# Read the integer N
N = int(sys.stdin.readline())

# Read the sequence A
# Using strip() to remove potential trailing newline characters from the input line
A = list(map(int, sys.stdin.readline().strip().split()))

# Check if there exists an index i (0-based) such that A[i], A[i+1], and A[i+2] are equal.
# The loop should iterate up to the last possible starting index for a triplet,
# which is N-3. Indices i, i+1, i+2 must all be within the list bounds [0, N-1].
# If i = N-3, then i+2 = (N-3)+2 = N-1, which is the last valid index.
# So, the loop should run for i in range(0, N-2).
found_triplet = False
for i in range(N - 2):
    # Check if the element at index i, i+1, and i+2 are all the same
    if A[i] == A[i+1] == A[i+2]:
        found_triplet = True
        break # Found a triplet, no need to check further

# Output the result based on whether a triplet was found
if found_triplet:
    print("Yes")
else:
    print("No")