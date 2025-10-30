import sys
from collections import deque

def bfs(start, adj, N):
    dist = [-1] * (N + 1)
    q = deque()
    dist[start] = 0
    q.append(start)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    M = int(input[ptr]); ptr +=1
    S = int(input[ptr]); ptr +=1
    T = int(input[ptr]); ptr +=1
    
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        u = int(input[ptr]); ptr +=1
        v = int(input[ptr]); ptr +=1
        adj[u].append(v)
        adj[v].append(u)
    
    dS = bfs(S, adj, N)
    dT = bfs(T, adj, N)
    d = dS[T]
    
    if d == 1:
        # Check if there exists a third node in S or T's adjacency
        other = set()
        for u in adj[S]:
            if u != T:
                other.add(u)
        for u in adj[T]:
            if u != S:
                other.add(u)
        if len(other) >= 1:
            print(3)
        else:
            print(-1)
        return
    
    # For d >= 2
    exists = False
    for u in range(1, N+1):
        if u == S or u == T:
            continue
        if dS[u] + dT[u] == d:
            exists = True
            break
    if exists:
        print(2 * d)
    else:
        # Check if there exists a path through two different nodes
        # Alternative approach: if S and T are part of a cycle
        # But not handled here
        print(-1)
    
if __name__ == "__main__":
    main()