import sys
import threading
from collections import deque

def main():
    import sys
    input = sys.stdin.readline
    
    N1, N2, M = map(int, input().split())
    N = N1 + N2
    
    # Build adjacency list for the undirected graph
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a, b = map(int, input().split())
        adj[a].append(b)
        adj[b].append(a)
    
    # BFS from node 1 to find distances within the first component
    dist1 = [-1] * (N+1)
    dq = deque([1])
    dist1[1] = 0
    while dq:
        u = dq.popleft()
        for w in adj[u]:
            if dist1[w] == -1:
                dist1[w] = dist1[u] + 1
                dq.append(w)
    # maximum distance from 1 to any node in [1..N1]
    max1 = max(dist1[1:N1+1])
    
    # BFS from the target node N = N1+N2 to find distances in the second component
    T = N
    dist2 = [-1] * (N+1)
    dq = deque([T])
    dist2[T] = 0
    while dq:
        u = dq.popleft()
        for w in adj[u]:
            if dist2[w] == -1:
                dist2[w] = dist2[u] + 1
                dq.append(w)
    # maximum distance from N to any node in [N1+1..N]
    max2 = max(dist2[N1+1:N+1])
    
    # By adding the edge between the farthest in comp1 and farthest in comp2,
    # the resulting distance is max1 + 1 + max2
    print(max1 + 1 + max2)

if __name__ == "__main__":
    main()