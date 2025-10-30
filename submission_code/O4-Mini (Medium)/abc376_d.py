import sys
from collections import deque

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    adj = [[] for _ in range(N+1)]
    in_edges_to_1 = []  # list of nodes u with an edge u->1

    for _ in range(M):
        a, b = map(int, input().split())
        adj[a].append(b)
        if b == 1:
            in_edges_to_1.append(a)

    # BFS from node 1 to find shortest distance d1[v] from 1 to v
    d1 = [-1] * (N+1)
    dq = deque([1])
    d1[1] = 0
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if d1[v] == -1:
                d1[v] = d1[u] + 1
                dq.append(v)

    # Among all edges u->1, pick the one minimizing d1[u] + 1
    ans = float('inf')
    for u in in_edges_to_1:
        if d1[u] != -1:
            ans = min(ans, d1[u] + 1)

    if ans == float('inf'):
        print(-1)
    else:
        print(ans)

if __name__ == "__main__":
    main()