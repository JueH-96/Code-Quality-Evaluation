# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

found_triplet = False

# We need to check for A[j], A[j+1], A[j+2] being equal.
# The index j (0-based) can range from 0 to N-3.
# The loop range(N - 2) generates integers 0, 1, ..., N-3.
# This is valid because N >= 3 as per constraints.
# For N=3, range(1) gives j=0. We check A[0], A[1], A[2].
# For N=4, range(2) gives j=0,1. We check (A[0],A[1],A[2]) and (A[1],A[2],A[3]).
for j in range(N - 2):
    if A[j] == A[j+1] and A[j+1] == A[j+2]:
        found_triplet = True
        break  # Triplet found, no need to continue searching

if found_triplet:
    print("Yes")
else:
    print("No")