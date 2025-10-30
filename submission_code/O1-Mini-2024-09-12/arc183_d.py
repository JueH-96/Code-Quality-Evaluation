# YOUR CODE HERE
import sys
import sys
import sys
from collections import deque

def readints():
    import sys
    return list(map(int, sys.stdin.read().split()))

def bfs(start, adj, N):
    depth = [ -1 ] * (N + 1)
    q = deque()
    q.append(start)
    depth[start] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                q.append(v)
    farthest_node = start
    max_depth = 0
    for i in range(1, N+1):
        if depth[i] > max_depth:
            max_depth = depth[i]
            farthest_node = i
    return farthest_node, depth

def main():
    data = readints()
    ptr = 0
    N = data[ptr]
    ptr +=1
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        A = data[ptr]
        B = data[ptr+1]
        adj[A].append(B)
        adj[B].append(A)
        ptr +=2
    # First BFS to find one end of the diameter
    u, _ = bfs(1, adj, N)
    # Second BFS to find the actual diameter and depths
    v, depth = bfs(u, adj, N)
    # Find all leaves
    leaves = []
    for i in range(1, N+1):
        if len(adj[i]) ==1:
            leaves.append((depth[i], i))
    # Sort leaves by depth
    leaves.sort()
    # Pair first with last, second with second last, etc.
    result = []
    i =0
    j = len(leaves)-1
    while i < j:
        result.append((leaves[i][1], leaves[j][1]))
        i +=1
        j -=1
    # If there's an odd number of leaves, pair the middle one with itself
    # But since N is even and tree has perfect matching, number of leaves should be even
    for pair in result:
        print(pair[0], pair[1])

if __name__ == "__main__":
    main()