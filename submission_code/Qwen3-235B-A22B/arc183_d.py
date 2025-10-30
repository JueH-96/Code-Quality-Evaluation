import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1

    from collections import defaultdict, deque

    adj = [[] for _ in range(N+1)]
    pair = lambda x: x ^ 1 if x % 2 == 1 else x ^ 1

    initial_edges = N//2
    edges = []
    for _ in range(N-1):
        a = int(input[ptr])
        b = int(input[ptr+1])
        ptr +=2
        adj[a].append(b)
        adj[b].append(a)
        if _ < initial_edges:
            edges.append( (a,b) )

    degree = [0]*(N+1)
    for u in range(1, N+1):
        degree[u] = len(adj[u])

    parent = [0]*(N+1)
    visited = [False]*(N+1)
    q = deque()
    q.append(1)
    visited[1] = True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v] and v != pair(u):
                visited[v] = True
                parent[v] = u
                q.append(v)

    ans = []
    leaves = deque()

    def is_leaf(u):
        return degree[u] == 1

    for u in range(1, N+1):
        if is_leaf(u):
            leaves.append(u)

    matched = [False]*(N+1)

    def get_leaf_other_than(u, p):
        for v in adj[u]:
            if v != p and not matched[v] and degree[v] == 1:
                return v
        return None

    # Process all nodes in BFS order to pair leaves
    q = deque()
    visited = [False]*(N+1)
    for u in range(1, N+1):
        if is_leaf(u):
            q.append(u)
            visited[u] = True

    while q:
        u = q.popleft()
        if matched[u]:
            continue
        v = pair(u)
        if matched[v]:
            # Find another leaf connected through BFS
            x = get_leaf_other_than(parent[u], u)
            if x:
                ans.append( (u, x) )
                matched[u] = matched[x] = True
                # Update degrees and parents
                degree[parent[u]] -= 1
                if degree[parent[u]] == 1:
                    q.append(parent[u])
                degree[parent[x]] -= 1
                if degree[parent[x]] == 1:
                    q.append(parent[x])
            continue
        # Check if v is also a leaf
        if is_leaf(v) and not matched[v]:
            ans.append( (u, v) )
            matched[u] = matched[v] = True
            # Update parents
            pu, pv = parent[u], parent[v]
            degree[pu] -= 1
            degree[pv] -= 1
            if degree[pu] == 1:
                q.append(pu)
            if degree[pv] == 1 and pu != pv:
                q.append(pv)
        else:
            # Try to find another leaf in the subtree
            # Not implemented
            pass

    for a, b in ans:
        print(a, b)

if __name__ == "__main__":
    main()