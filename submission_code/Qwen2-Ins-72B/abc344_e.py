from collections import OrderedDict

N = int(input())
A = list(map(int, input().split()))
Q = int(input())

d = OrderedDict()
for i, a in enumerate(A):
    d[a] = i

for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        x, y = query[1], query[2]
        idx = d[x]
        A.insert(idx + 1, y)
        d[y] = idx + 1
    else:
        x = query[1]
        idx = d[x]
        del d[x]
        A.pop(idx)

print(*A)