# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    parent = [i for i in range(N+1)]
    size = [1] * (N+1)
    x_list = [[] for _ in range(N+1)]
    y_list = [[] for _ in range(N+1)]
    def find(u):
        while u != parent[u]:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        u = find(u)
        v = find(v)
        if u == v:
            return
        if size[u] < size[v]:
            u, v = v, u
        parent[v] = u
        size[u] += size[v]
    edges = []
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        edges.append((u, v, 0))
    K = int(sys.stdin.readline())
    x_pairs = []
    x_pairs_indices = {}
    for idx in range(K):
        x_i, y_i = map(int, sys.stdin.readline().split())
        x_pairs.append((x_i, y_i, idx))
        # Note that x_i and y_i are not connected at time 0
    Q = int(sys.stdin.readline())
    query_edges = []
    for t in range(1, Q+1):
        u, v = map(int, sys.stdin.readline().split())
        edges.append((u, v, t))
    # Union-Find with tracking of x_i - y_i pairs
    # Initialize per component x_set and y_set
    NODES = N + 1
    parent = [i for i in range(NODES)]
    size = [1] * NODES
    x_set = [set() for _ in range(NODES)]
    y_set = [set() for _ in range(NODES)]
    # For each x_i, y_i pair, store indices
    for idx, (x_i, y_i, _) in enumerate(x_pairs):
        x_set[x_i].add(idx)
        y_set[y_i].add(idx)
    # Initialize time of connectivity for each pair to infinity
    t_i = [Q+1]*K
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return parent[u]
    def union(u, v, time):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return
        # Before merging, check for overlapping pairs
        # Always merge smaller into larger
        if size[u_root] < size[v_root]:
            u_root, v_root = v_root, u_root
        # Now, u_root is larger
        # Check for indices where x is in u_root and y is in v_root
        common_indices = x_set[u_root].intersection(y_set[v_root])
        for idx in common_indices:
            if t_i[idx] > time:
                t_i[idx] = time
        common_indices = x_set[v_root].intersection(y_set[u_root])
        for idx in common_indices:
            if t_i[idx] > time:
                t_i[idx] = time
        # Merge sets
        x_set[u_root].update(x_set[v_root])
        y_set[u_root].update(y_set[v_root])
        # Update parent and size
        parent[v_root] = u_root
        size[u_root] += size[v_root]
    # First, process initial edges (time=0)
    for u, v, t in edges:
        if t == 0:
            union(u, v, t)
    # Now process query edges in order
    for u, v, t in edges:
        if t > 0:
            union(u, v, t)
    # Now, t_i contains the earliest time when each x_i - y_i pair becomes connected
    # Build cnt array
    Qp2 = Q + 2
    cnt = [0]*Qp2
    for idx, time in enumerate(t_i):
        if time <= Q:
            cnt[time] += 1
    # Compute prefix sums
    ans = ['Yes'] * (Q+1)  # ans[1..Q]
    total = 0
    for t in range(1, Q+1):
        total += cnt[t]
        if total > 0:
            ans[t] = 'No'
    # Output the answers
    for t in range(1, Q+1):
        print(ans[t])

threading.Thread(target=main).start()