# Read the two integers A and B from input
A, B = map(int, input().split())

# Check for horizontal adjacency based on the derived conditions
# 1. B must be A + 1 (since A < B is given, and they must be next to each other)
# 2. A must not be in the rightmost column (i.e., A is not 3 or 6).
#    A number is in the rightmost column if it's a multiple of 3.
#    So, A % 3 should not be 0.

if B == A + 1 and A % 3 != 0:
    print("Yes")
else:
    print("No")