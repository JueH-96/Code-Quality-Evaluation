# YOUR CODE HERE
import sys
from itertools import product

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
M = int(data[N+1])
B = list(map(int, data[N+2:N+2+M]))
L = int(data[N+2+M])
C = list(map(int, data[N+3+M:N+3+M+L]))
Q = int(data[N+3+M+L])
X = list(map(int, data[N+4+M+L:N+4+M+L+Q]))

# Precompute all possible sums of elements from A and B
AB_sums = set(a + b for a, b in product(A, B))

# Check if any of the sums in X can be formed by adding an element from C
results = []
for x in X:
    if any(x - c in AB_sums for c in C):
        results.append("Yes")
    else:
        results.append("No")

print("
".join(results))