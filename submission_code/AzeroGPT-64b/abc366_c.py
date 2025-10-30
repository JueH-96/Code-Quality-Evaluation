import sys
from collections import defaultdict

Q = int(input())

cnts = defaultdict(int)
uniq_cnt = 0

queries = []
for _ in range(Q):
    queries.append(list(map(int, input().split())))

res = []
for q in queries:
    if q[0] == 1:
        cnts[q[1]] += 1

        if cnts[q[1]] == 1:
            uniq_cnt += 1
    elif q[0] == 2:
        cnts[q[1]] -= 1

        if cnts[q[1]] == 0:
            uniq_cnt -= 1
    elif q[0] == 3:
        res.append(uniq_cnt)

print('
'.join(map(str, res)))