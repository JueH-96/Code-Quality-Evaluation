import sys
from collections import deque
from heapq import heapify,heappop,heappush

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    A.append(A[0])
    B.append(B[0])
    G = [deque() for _ in range(M)]
    for i in range(N):
        G[A[i]].append(i)
    for i in range(M):
        G[i].append(N)
    dist = [10**18]*N
    que = []
    for i in range(N):
        while G[B[i]][0] < i:
            G[B[i]].popleft()
        if G[B[i]][0] == i:
            dist[i] = 0
            heappush(que, (0, i))
        elif i and B[i-1] == B[i]:
            dist[i] = dist[i-1]
            heappush(que, (dist[i], i))
    while que:
        d, v = heappop(que)
        if dist[v] < d:
            continue
        for nv in [v-1, v+1]:
            if nv < 0 or nv >= N:
                continue
            nd = d + abs(B[nv]-B[v])
            if B[nv] == B[v]:
                nd -= M-1
            if dist[nv] > nd:
                dist[nv] = nd
                heappush(que, (nd, nv))
    if min(dist) < 0:
        print(-1)
    else:
        print(sum(dist))

if __name__ == "__main__":
    main()