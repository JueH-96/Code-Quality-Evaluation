from collections import deque

def solve():
    N = int(input())
    indeg = [0] * N
    edges = [[] for _ in range(N)]
    for i in range(N):
        C, *P = map(int, input().split())
        for p in P:
            p -= 1
            edges[p].append(i)
            indeg[i] += 1

    q = deque([i for i in range(N) if indeg[i] == 0])
    ans = []
    while q:
        i = q.popleft()
        if i != 0:
            ans.append(i)
        for j in edges[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                q.append(j)
    print(' '.join(map(str, ans)))

solve()