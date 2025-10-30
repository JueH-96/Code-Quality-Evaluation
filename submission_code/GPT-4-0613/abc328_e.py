import sys
from itertools import combinations
from operator import itemgetter

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    edges = [list(map(int, sys.stdin.readline().split())) for _ in range(M)]
    edges.sort(key=itemgetter(2))

    ans = K
    for bit in range(1 << N):
        if bin(bit).count('1') != N - 1:
            continue
        used = [0] * N
        for i in range(N):
            if bit & (1 << i):
                used[i] = 1
        cost = 0
        for u, v, w in edges:
            if used[u - 1] ^ used[v - 1]:
                cost += w
                cost %= K
                used[u - 1] = used[v - 1] = 1
        if sum(used) == N:
            ans = min(ans, cost)
    print(ans)

solve()