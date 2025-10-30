import sys
from itertools import product

N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))
M = int(sys.stdin.readline().strip())
B = list(map(int, sys.stdin.readline().strip().split()))
L = int(sys.stdin.readline().strip())
C = list(map(int, sys.stdin.readline().strip().split()))
Q = int(sys.stdin.readline().strip())
X = list(map(int, sys.stdin.readline().strip().split()))

# Generate all possible combinations of elements from A, B, and C
combinations = list(product(A, B, C))

# Create a set for quick lookup
combinations_set = set(combinations)

for x in X:
    if (x, x, x) in combinations_set:
        print("Yes")
    else:
        print("No")