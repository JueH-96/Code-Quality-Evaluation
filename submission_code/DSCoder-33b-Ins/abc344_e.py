# YOUR CODE HERE
from collections import deque

N = int(input())
A = deque(map(int, input().split()))
Q = int(input())

for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        x, y = query[1], query[2]
        A.insert(A.index(x) + 1, y)
    else:
        x = query[1]
        A.remove(x)

print(*A)