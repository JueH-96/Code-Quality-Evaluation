import sys

# Read input N
N = int(sys.stdin.readline())

# Read input sequence A as a list of integers
A = list(map(int, sys.stdin.readline().split()))

# Flag to indicate if a triplet is found
found_triplet = False

# Iterate through the sequence A
# We need to check elements at index i, i+1, and i+2.
# The loop should go up to N-3 (inclusive) for index i,
# so that i+2 does not go out of bounds (which would be index N-1).
# range(N - 2) generates indices from 0 to N-3.
for i in range(N - 2):
    # Check if the elements at indices i, i+1, and i+2 are equal
    if A[i] == A[i+1] and A[i+1] == A[i+2]:
        # If they are equal, set the flag to True
        found_triplet = True
        # We found a triplet, no need to check further, so break the loop
        break

# After checking the entire sequence, print the result based on the flag
if found_triplet:
    print("Yes")
else:
    print("No")