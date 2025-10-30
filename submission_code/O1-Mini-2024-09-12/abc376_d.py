# YOUR CODE HERE
import sys
import sys
import sys
from collections import deque

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    edges = data[2:]
    
    adj_rev = [[] for _ in range(N+1)]
    outgoing_from_1 = []
    
    for i in range(M):
        a = int(edges[2*i])
        b = int(edges[2*i+1])
        if a == 1:
            outgoing_from_1.append(b)
        adj_rev[b].append(a)
    
    # BFS on reversed graph from node 1
    dist = [-1] * (N +1)
    q = deque()
    dist[1] = 0
    q.append(1)
    while q:
        u = q.popleft()
        for v in adj_rev[u]:
            if dist[v] == -1:
                dist[v] = dist[u] +1
                q.append(v)
    
    min_cycle = float('inf')
    for v in outgoing_from_1:
        if dist[v] != -1:
            cycle_length = 1 + dist[v]
            if cycle_length < min_cycle:
                min_cycle = cycle_length
    
    if min_cycle != float('inf'):
        print(min_cycle)
    else:
        print(-1)

if __name__ == "__main__":
    main()