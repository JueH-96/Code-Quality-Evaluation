import sys
from collections import deque

def main():
    N1, N2, M = map(int, sys.stdin.readline().split())
    edges = [list(map(int, sys.stdin.readline().split())) for _ in range(M)]
    G = [set() for _ in range(N1+N2+1)]
    for a, b in edges:
        G[a].add(b)
        G[b].add(a)
    dist1 = [-1]*(N1+N2+1)
    distN = [-1]*(N1+N2+1)
    dist1[1] = 0
    distN[N1+N2] = 0
    que = deque([1])
    while que:
        v = que.popleft()
        for nv in G[v]:
            if dist1[nv] == -1:
                dist1[nv] = dist1[v] + 1
                que.append(nv)
    que = deque([N1+N2])
    while que:
        v = que.popleft()
        for nv in G[v]:
            if distN[nv] == -1:
                distN[nv] = distN[v] + 1
                que.append(nv)
    ans = 0
    for i in range(1, N1+1):
        if dist1[i] != -1 and distN[i] != -1:
            ans = max(ans, dist1[i] + distN[i] + 1)
    print(ans)

if __name__ == "__main__":
    main()