# YOUR CODE HERE
import sys
input = sys.stdin.read
from itertools import product

data = input().split()
index = 0

N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))
index += N

M = int(data[index])
index += 1
B = list(map(int, data[index:index + M]))
index += M

L = int(data[index])
index += 1
C = list(map(int, data[index:index + L]))
index += L

Q = int(data[index])
index += 1
X = list(map(int, data[index:index + Q]))

# Precompute all possible sums of elements from A and B
AB_sums = set(a + b for a, b in product(A, B))

# For each query, check if there exists a sum in AB_sums that, when added to an element in C, equals the query value
results = []
for x in X:
    found = False
    for c in C:
        if (x - c) in AB_sums:
            found = True
            break
    if found:
        results.append("Yes")
    else:
        results.append("No")

# Print the results
print("
".join(results))