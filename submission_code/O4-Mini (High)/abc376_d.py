import sys
from collections import deque

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    # Build the reversed graph: adj_rev[b] contains all a such that a->b in original
    adj_rev = [[] for _ in range(N+1)]
    # Keep track of all direct neighbors v of 1 (edges 1->v in the original graph)
    outgoing_1 = []
    
    for _ in range(M):
        a, b = map(int, input().split())
        adj_rev[b].append(a)
        if a == 1:
            outgoing_1.append(b)
    
    # dist[u] = shortest distance from u to 1 in the original graph, computed
    # by BFS on the reversed graph starting from 1.
    dist = [-1] * (N+1)
    q = deque([1])
    dist[1] = 0
    
    while q:
        u = q.popleft()
        for v in adj_rev[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    
    # For each edge 1->v, if there's a path from v back to 1 of length dist[v],
    # then that forms a cycle of length 1 + dist[v].
    ans = float('inf')
    for v in outgoing_1:
        if dist[v] != -1:
            ans = min(ans, dist[v] + 1)
    
    print(ans if ans != float('inf') else -1)

if __name__ == '__main__':
    main()