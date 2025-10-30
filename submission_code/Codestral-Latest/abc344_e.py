import sys
from collections import deque

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
A = deque(int(data[i]) for i in range(index, index + N))
index += N
Q = int(data[index])
index += 1

element_to_index = {x: i for i, x in enumerate(A)}

for _ in range(Q):
    query_type = int(data[index])
    index += 1
    if query_type == 1:
        x = int(data[index])
        y = int(data[index + 1])
        index += 2
        pos = element_to_index[x]
        A.insert(pos + 1, y)
        element_to_index[y] = pos + 1
        for i in range(pos + 2, len(A)):
            element_to_index[A[i]] += 1
    elif query_type == 2:
        x = int(data[index])
        index += 1
        pos = element_to_index[x]
        del A[pos]
        del element_to_index[x]
        for i in range(pos, len(A)):
            element_to_index[A[i]] -= 1

print(" ".join(map(str, A)))