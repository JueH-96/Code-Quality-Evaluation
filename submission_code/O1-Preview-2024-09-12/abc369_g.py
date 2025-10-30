# YOUR CODE HERE
import sys
import threading

sys.setrecursionlimit(1 << 25)

def main():
    N = int(sys.stdin.readline())
    tree = [[] for _ in range(N)]
    for _ in range(N - 1):
        u, v, l = map(int, sys.stdin.readline().split())
        u -= 1  # Convert to 0-based indexing
        v -= 1
        tree[u].append((v, l))
        tree[v].append((u, l))

    # Perform BFS from node 0 to compute distances and parents
    from collections import deque
    q = deque()
    q.append(0)
    dist = [0] * N
    parent = [-1] * N
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v, w in tree[u]:
            if v != parent[u]:
                dist[v] = dist[u] + w
                parent[v] = u
                q.append(v)

    # Define net contribution for each node (only leaves contribute)
    net_contrib = []
    degrees = [len(tree[u]) for u in range(N)]
    for u in range(1, N):
        if degrees[u] == 1:
            net = dist[u]
            net_contrib.append((net, u))

    # Sort nodes by net contribution (distance from root) in decreasing order
    net_contrib.sort(reverse=True)
    # Keep track of included edges
    included = set()
    total_weight = 0
    ans = []
    idx = 0  # Index in net_contrib list
    K = 1
    ans_list = [0] * N
    max_total_weight = 0
    included_edges = set()

    # Mapping edges to unique ids
    edge_id = {}
    edge_counter = 0
    for u in range(N):
        for v, _ in tree[u]:
            if (u, v) not in edge_id and (v, u) not in edge_id:
                edge_id[(u, v)] = edge_counter
                edge_id[(v, u)] = edge_counter
                edge_counter +=1

    # The initial total weight is zero
    total_weight = 0

    # Go through the nodes with positive net contribution
    included = [False] * edge_counter
    for K in range(1, N+1):
        if idx < len(net_contrib):
            net, u = net_contrib[idx]
            idx += 1
            x = u
            while x != 0:
                e_id = edge_id[(x, parent[x])]
                if not included[e_id]:
                    for v, w in tree[x]:
                        if v == parent[x]:
                            total_weight += w
                            included[e_id] = True
                            break
                x = parent[x]
        ans_list[K-1] = total_weight * 2

    # For K beyond the number of leaves, the total weight remains the same
    for K in range(idx+1, N+1):
        ans_list[K-1] = total_weight *2

    for ans in ans_list:
        print(ans)