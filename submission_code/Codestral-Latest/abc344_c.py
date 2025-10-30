# YOUR CODE HERE
import sys
input = sys.stdin.read
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

# Precompute all possible sums of two elements from B and C
sums_BC = set()
for b in B:
    for c in C:
        sums_BC.add(b + c)

# For each X_i, check if there exists an A_j such that A_j + sum_BC = X_i
for x in X:
    found = False
    for a in A:
        if x - a in sums_BC:
            found = True
            break
    if found:
        print("Yes")
    else:
        print("No")