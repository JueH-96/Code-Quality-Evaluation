# YOUR CODE HERE
import sys
import threading

import sys
import sys
# For recursive limit
sys.setrecursionlimit(1 << 25)
def main():
    import sys
    import threading
    N,*rest = list(map(int, sys.stdin.read().split()))
    N = int(N)
    adj = [[] for _ in range(N+1)]
    degree = [0]*(N+1)
    edges = []
    idx = 0
    for _ in range(N-1):
        A = rest[idx]
        B = rest[idx+1]
        idx +=2
        adj[A].append(B)
        adj[B].append(A)
        degree[A] +=1
        degree[B] +=1
        edges.append((A,B))

    depth = [0]*(N+1)
    matches = []
    total_score = 0

    def dfs(u, p):
        Leaves = []
        for v in adj[u]:
            if v != p:
                depth[v] = depth[u]+1
                Lv = dfs(v, u)
                Leaves.extend(Lv)
        if len(adj[u]) == 1 and u != p:
            # u is a leaf
            Leaves.append(u)
        while len(Leaves) >=2:
            l1 = Leaves.pop()
            l2 = Leaves.pop()
            dist = depth[l1]+depth[l2]-2*depth[u]
            matches.append((l1,l2))
        # Return any unmatched leaf
        return Leaves

    root = 1
    depth[root] = 0
    unLeaves = dfs(root, -1)

    if unLeaves:
        # If unmatched leaves remain at root, process them
        # If root is a leaf, include it
        if len(adj[root]) == 0 or (len(adj[root]) == 1 and root != 1):
            unLeaves.append(root)
        while len(unLeaves) >= 2:
            l1 = unLeaves.pop()
            l2 = unLeaves.pop()
            dist = depth[l1]+depth[l2]-2*depth[root]
            matches.append((l1,l2))

    # Output the matches
    for x,y in matches:
        print(f"{x} {y}")