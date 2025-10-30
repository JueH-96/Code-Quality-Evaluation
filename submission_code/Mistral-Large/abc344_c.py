import sys
from itertools import product

# Read input
N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))
M = int(sys.stdin.readline().strip())
B = list(map(int, sys.stdin.readline().strip().split()))
L = int(sys.stdin.readline().strip())
C = list(map(int, sys.stdin.readline().strip().split()))
Q = int(sys.stdin.readline().strip())
X = list(map(int, sys.stdin.readline().strip().split()))

# Generate all possible sums of one element from each of A, B, and C
possible_sums = set(a + b + c for a, b, c in product(A, B, C))

# Check each X_i and print Yes or No
for x in X:
    if x in possible_sums:
        print("Yes")
    else:
        print("No")