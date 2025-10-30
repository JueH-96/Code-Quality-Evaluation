import sys

# Read N, A
N = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split()))

# Read M, B
M = int(sys.stdin.readline())
B = list(map(int, sys.stdin.readline().split()))

# Read L, C
L = int(sys.stdin.readline())
C = list(map(int, sys.stdin.readline().split()))

# Read Q, X
Q = int(sys.stdin.readline())
X = list(map(int, sys.stdin.readline().split()))

# Precompute all possible sums of elements from A and B
# Store them in a set for efficient O(1) average time lookup
sum_ab = set()
for a in A:
    for b in B:
        sum_ab.add(a + b)

# For each query X_i, check if it is possible to find a, b, c such that a + b + c = X_i
# This is equivalent to checking if X_i - c exists in the precomputed set sum_ab
for x in X:
    found = False
    # Iterate through each element c in C
    for c in C:
        # Calculate the required sum from A and B
        required_ab = x - c
        
        # Check if the required sum (required_ab) is present in the set sum_ab
        # Since all elements in A and B are non-negative, all sums in sum_ab are non-negative.
        # If required_ab is negative, it cannot be in sum_ab, so the lookup will correctly return False.
        # We don't need an explicit check for required_ab >= 0 before the set lookup.
        if required_ab in sum_ab:
            found = True
            break # Found a valid combination (a+b=required_ab, c), no need to check other elements in C for this x
    
    # Print the result for the current query X_i
    if found:
        print("Yes")
    else:
        print("No")