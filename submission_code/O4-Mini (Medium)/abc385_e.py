#!/usr/bin/env python3
import sys
import threading

def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    if not line:
        return
    N = int(line[0])
    adj = [[] for _ in range(N)]
    for _ in range(N-1):
        u,v = data.readline().split()
        u = int(u)-1; v = int(v)-1
        adj[u].append(v)
        adj[v].append(u)
    deg = [len(adj[i]) for i in range(N)]
    global_max = 0
    # For each node as potential center
    for u in range(N):
        d = deg[u]
        if d == 0:
            continue
        # Build list of leaf-availability for each neighbor
        L = []
        # L_j = number of neighbors of v other than u = deg[v] - 1
        for v in adj[u]:
            # deg[v]-1 might be zero
            L.append(deg[v] - 1)
        # sort descending
        L.sort(reverse=True)
        best_u = 0
        # Try x = 1..len(L)
        # y = L[x-1], require y >= 1
        # nodes_contrib = x * (y + 1)
        for i, y in enumerate(L, start=1):
            if y <= 0:
                break
            val = i * (y + 1)
            if val > best_u:
                best_u = val
        if best_u > global_max:
            global_max = best_u
    # The best snowflake size is 1 + global_max
    best_size = 1 + global_max
    # We must delete N - best_size vertices
    print(N - best_size)

if __name__ == "__main__":
    main()