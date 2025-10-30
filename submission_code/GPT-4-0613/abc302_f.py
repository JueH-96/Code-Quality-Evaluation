import sys
from collections import deque
input = sys.stdin.readline

N, M = map(int, input().split())
G = [[] for _ in range(M+1)]
for _ in range(N):
    a = int(input())
    S = list(map(int, input().split()))
    for s in S:
        G[s].append(S)

dist = [-1]*(M+1)
dist[1] = 0
que = deque([1])

while que:
    v = que.popleft()
    while G[v]:
        S = G[v].pop()
        for s in S:
            if dist[s] == -1:
                dist[s] = dist[v] + 1
                que.append(s)

print(dist[M])