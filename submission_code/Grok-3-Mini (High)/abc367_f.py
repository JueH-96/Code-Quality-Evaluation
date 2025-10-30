import sys
from collections import Counter

data = sys.stdin.read().split()
index = 0
N, Q = int(data[index]), int(data[index + 1])
index += 2
A = list(map(int, data[index:index + N]))
index += N
B = list(map(int, data[index:index + N]))
index += N

for _ in range(Q):
    l = int(data[index])
    r = int(data[index + 1])
    L = int(data[index + 2])
    R = int(data[index + 3])
    index += 4

    if (r - l + 1) != (R - L + 1):
        print("No")
        continue

    # Count for A subarray
    count_A = Counter()
    for i in range(l - 1, r):  # l-1 to r-1 inclusive (0-based indexing)
        count_A[A[i]] += 1

    # Count for B subarray
    count_B = Counter()
    for j in range(L - 1, R):  # L-1 to R-1 inclusive (0-based indexing)
        count_B[B[j]] += 1

    if count_A == count_B:
        print("Yes")
    else:
        print("No")