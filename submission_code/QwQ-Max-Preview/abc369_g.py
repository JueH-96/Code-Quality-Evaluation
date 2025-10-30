import sys
import heapq

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]  # 1-based indexing

    for _ in range(N-1):
        u, v, l = map(int, sys.stdin.readline().split())
        edges[u].append((v, l))
        edges[v].append((u, l))

    # Compute distances from root (1)
    from collections import deque
    dist = [0] * (N + 1)
    parent = [0] * (N + 1)
    visited = [False] * (N + 1)
    q = deque([1])
    visited[1] = True

    while q:
        u = q.popleft()
        for v, l in edges[u]:
            if not visited[v] and v != parent[u]:
                parent[v] = u
                dist[v] = dist[u] + l
                visited[v] = True
                q.append(v)

    # Decompose the tree into chains
    chains = []
    visited_chain = [False] * (N + 1)

    def dfs(u):
        chain = []
        current = u
        while True:
            chain.append(current)
            visited_chain[current] = True
            children = []
            for v, l in edges[current]:
                if v != parent[current] and not visited_chain[v]:
                    children.append(v)
            if len(children) != 1:
                break
            current = children[0]
        # Extract the distances and sort in descending order
        chain_dists = [dist[node] for node in chain]
        chain_dists.sort(reverse=True)
        return chain_dists

    # Start from the root's children
    for v, l in edges[1]:
        if v != parent[1] and not visited_chain[v]:
            chain = dfs(v)
            if chain:
                chains.append(chain)

    # Check for any remaining nodes (leaves that are not processed)
    for u in range(2, N + 1):
        if not visited_chain[u]:
            chain = dfs(u)
            if chain:
                chains.append(chain)

    # Now, collect all chains and prepare for merging
    heap = []
    for chain in chains:
        if chain:
            heapq.heappush(heap, (-chain[0], 0, chain))

    merged = []
    while heap:
        val, idx, chain = heapq.heappop(heap)
        val = -val
        merged.append(val)
        if idx + 1 < len(chain):
            heapq.heappush(heap, (-chain[idx + 1], idx + 1, chain))

    # Compute sum of all edges
    sum_edges = 0
    for u in range(1, N + 1):
        for v, l in edges[u]:
            if parent[v] == u:  # u is parent of v, so edge is u->v
                sum_edges += l

    # Prepare prefix sums
    prefix = [0] * (len(merged) + 1)
    for i in range(len(merged)):
        prefix[i + 1] = prefix[i] + merged[i]

    # Process each K from 1 to N
    for K in range(1, N + 1):
        if K <= len(merged):
            ans = 2 * prefix[K]
        else:
            ans = 2 * sum_edges
        print(ans)

if __name__ == "__main__":
    main()