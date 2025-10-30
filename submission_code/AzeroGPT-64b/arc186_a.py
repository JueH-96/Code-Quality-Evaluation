import sys
from itertools import starmap

MOD = 1_000_000_007

def f(N, K):
    raw = [1 << i + j for i in range(N-1) for j in range(N)] + [1 << (N-1)*N + i for i in range(N)] + [2**((N-1)*N + i) for i in range(N)]
    mx = sum(raw)
    res = set([mx])
    for a in raw:
        res |= {x ^ a for x in res}
    return int(mx - K) in res

N, Q = map(int, input().split())
print('
'.join(starmap(lambda K: 'Yes' if f(N, K) else 'No', (map(int, (None for _ in range(Q)))))))