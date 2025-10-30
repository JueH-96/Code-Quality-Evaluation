import bisect

n, X = map(int, input().split())
A = list(map(int, input().split()))

from collections import defaultdict
index_map = defaultdict(list)
for idx, num in enumerate(A):
    index_map[num].append(idx)

found = False
for i in range(n):
    for j in range(i + 1, min(n, i + 200)):
        target = X - A[i] - A[j]
        if target not in index_map:
            continue
        indices = index_map[target]
        pos = bisect.bisect_right(indices, j)
        if pos < len(indices):
            k = indices[pos]
            print(i + 1, j + 1, k + 1)
            found = True
            exit()
if not found:
    print(-1)