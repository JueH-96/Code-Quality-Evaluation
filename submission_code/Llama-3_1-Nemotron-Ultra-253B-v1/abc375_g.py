import sys
import heapq

def main():
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    M = int(input[ptr])
    ptr +=1

    edges = []
    original_adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a = int(input[ptr])
        ptr +=1
        b = int(input[ptr])
        ptr +=1
        c = int(input[ptr])
        ptr +=1
        edges.append( (a, b, c) )
        original_adj[a].append( (b, c) )
        original_adj[b].append( (a, c) )

    def dijkstra(start, adj):
        dist = [float('inf')] * (N+1)
        dist[start] = 0
        heap = [ (0, start) ]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, c in adj[u]:
                if dist[v] > d + c:
                    dist[v] = d + c
                    heapq.heappush(heap, (dist[v], v))
        return dist

    d1 = dijkstra(1, original_adj)
    dn = dijkstra(N, original_adj)
    D = d1[N]

    subgraph_edges = []
    is_in_subgraph = [False] * M
    for i in range(M):
        a, b, c = edges[i]
        if (d1[a] + c + dn[b] == D) or (d1[b] + c + dn[a] == D):
            subgraph_edges.append( (a, b) )
            is_in_subgraph[i] = True

    # Build adjacency list for subgraph
    adj_sub = [[] for _ in range(N+1)]
    for a, b in subgraph_edges:
        adj_sub[a].append(b)
        adj_sub[b].append(a)

    # Find bridges using iterative Tarjan's algorithm
    disc = [0] * (N+1)
    low = [0] * (N+1)
    visited = [False] * (N+1)
    bridges = set()
    time_counter = 1
    parent = [ -1 ] * (N+1)
    stack = []

    for u in range(1, N+1):
        if not visited[u]:
            stack.append( (u, False, -1) )
            while stack:
                node, processed, p = stack.pop()
                if not processed:
                    if visited[node]:
                        continue
                    visited[node] = True
                    disc[node] = low[node] = time_counter
                    time_counter +=1
                    stack.append( (node, True, p) )
                    # Push children in reverse order
                    for v in reversed(adj_sub[node]):
                        if v == p:
                            continue
                        if not visited[v]:
                            parent[v] = node
                            stack.append( (v, False, node) )
                        else:
                            low[node] = min(low[node], disc[v])
                else:
                    for v in adj_sub[node]:
                        if v == p:
                            continue
                        if parent[v] == node:
                            low[node] = min(low[node], low[v])
                            if low[v] > disc[node]:
                                bridges.add(frozenset( (node, v) ))
                        elif disc[v] < disc[node]:
                            low[node] = min(low[node], disc[v])

    # Build DSU for subgraph excluding bridges
    parent_dsu = list(range(N+1))
    def find_dsu(u):
        while parent_dsu[u] != u:
            parent_dsu[u] = parent_dsu[parent_dsu[u]]
            u = parent_dsu[u]
        return u

    def union_dsu(u, v):
        u_root = find_dsu(u)
        v_root = find_dsu(v)
        if u_root != v_root:
            parent_dsu[v_root] = u_root

    for a, b in subgraph_edges:
        if frozenset( (a, b) ) not in bridges:
            union_dsu(a, b)

    # Prepare root_1 and root_N
    root_1 = find_dsu(1)
    root_N = find_dsu(N)

    # Process each edge
    for i in range(M):
        a, b, c = edges[i]
        if not is_in_subgraph[i]:
            print("No")
        else:
            if frozenset( (a, b) ) in bridges:
                root_a = find_dsu(a)
                root_b = find_dsu(b)
                if (root_a == root_1 and root_b == root_N) or (root_b == root_1 and root_a == root_N):
                    print("Yes")
                else:
                    print("No")
            else:
                print("No")

if __name__ == '__main__':
    main()