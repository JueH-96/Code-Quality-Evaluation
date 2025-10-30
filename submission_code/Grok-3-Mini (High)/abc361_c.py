import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
K = int(data[index])
index += 1
A = [int(x) for x in data[index:index + N]]
Len = N - K
A.sort()
min_diff = min(A[i + Len - 1] - A[i] for i in range(N - Len + 1))
print(min_diff)