from collections import deque

def solve():
    N, M = map(int, input().split())
    S = [input() for _ in range(N)]

    G = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if sum(x != y for x, y in zip(S[i], S[j])) == 1:
                G[i].append(j)

    visited = [0] * N
    Q = deque([0])
    while Q:
        u = Q.popleft()
        for v in G[u]:
            if visited[v] == 0:
                visited[v] = 1
                Q.append(v)

    if sum(visited) == N:
        print('Yes')
    else:
        print('No')

solve()