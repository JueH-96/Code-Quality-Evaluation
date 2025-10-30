from collections import Counter
from math import inf

for _ in range(int(input())):
    n, k = map(int, input().split())
    s = input()

    c = Counter(s)
    o = 0
    for i in c.values():
        o += (i & 1)

    if o == 0:
        o = inf
    elif o < inf:
        o += 1

    print('YES' if o <= k + 2 else 'NO')