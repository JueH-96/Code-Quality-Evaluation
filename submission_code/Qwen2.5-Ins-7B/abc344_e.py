# YOUR CODE HERE
from collections import defaultdict

N = int(input())
A = list(map(int, input().split()))
Q = int(input())

pos = defaultdict(int)
for i, a in enumerate(A):
    pos[a] = i

insertions = defaultdict(list)
deletions = set()

for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        x, y = query[1], query[2]
        pos[y] = pos[x] + 1
        insertions[pos[x]].append((y, x))
    else:
        x = query[1]
        pos[x] = -1
        deletions.add(pos[x] + 1)

result = []
for i in range(N):
    if i in pos and pos[i] != -1 and pos[i] not in deletions:
        result.append(A[i])

for y, x in sorted(insertions.items()):
    if y not in deletions:
        result.insert(y, x[0])

print(*result)