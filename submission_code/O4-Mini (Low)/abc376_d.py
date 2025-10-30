import sys
import threading
from collections import deque

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    # adjacency list for reversed graph
    rev_adj = [[] for _ in range(N+1)]
    # list of direct neighbors from 1
    starts = []
    for _ in range(M):
        a, b = map(int, input().split())
        # store reversed edge b -> a
        rev_adj[b].append(a)
        # if it's an outgoing edge from 1, record b
        if a == 1:
            starts.append(b)
    # BFS on reversed graph from 1
    INF = 10**18
    dist = [INF] * (N+1)
    dq = deque([1])
    dist[1] = 0
    while dq:
        u = dq.popleft()
        for v in rev_adj[u]:
            if dist[v] == INF:
                dist[v] = dist[u] + 1
                dq.append(v)
    # For every edge 1->v, if v can reach 1, cycle length = dist[v] + 1
    ans = INF
    for v in starts:
        if dist[v] < INF:
            ans = min(ans, dist[v] + 1)
    print(ans if ans < INF else -1)

if __name__ == "__main__":
    main()