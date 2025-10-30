import sys
from collections import deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
H = list(map(int, data[1:]))

A = [0] * (N + 1)
result = [0] * N
operations = 0

queue = deque(range(N))

while queue:
    operations += 1
    A[0] += 1

    i = 0
    while i < N:
        if A[i] > A[i + 1] and A[i] > H[i]:
            A[i] -= 1
            A[i + 1] += 1
            if i in queue:
                queue.remove(i)
            if A[i + 1] > 0 and result[i] == 0:
                result[i] = operations
        else:
            i += 1

print(" ".join(map(str, result)))