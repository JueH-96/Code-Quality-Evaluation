from collections import deque

Q = int(input())
A = deque()
ans = []

for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        A.appendleft(query[1])
    else:
        k = query[1]
        ans.append(A[k-1])

for a in ans:
    print(a)