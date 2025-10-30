import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    original_pairs = []
    non_matching_edges = []
    node_to_pair = {}
    pair_edges = []
    for i in range(1, N//2 + 1):
        a, b = map(int, sys.stdin.readline().split())
        original_pairs.append((a, b))
        pair_edges.append((a, b))
        edges[a].append((b, i))
        edges[b].append((a, i))
        node_to_pair[a] = i-1
        node_to_pair[b] = i-1
    for _ in range(N-1 - N//2):
        a, b = map(int, sys.stdin.readline().split())
        non_matching_edges.append((a, b))

    has_non_matching = [False] * (N+1)
    for a, b in non_matching_edges:
        has_non_matching[a] = True
        has_non_matching[b] = True

    meta_adj = [[] for _ in range(N//2)]
    for a, b in non_matching_edges:
        pa = node_to_pair[a]
        pb = node_to_pair[b]
        meta_adj[pa].append(pb)
        meta_adj[pb].append(pa)

    leaf_nodes = []
    for i in range(N//2):
        u, v = original_pairs[i]
        if not has_non_matching[u]:
            leaf_nodes.append((u, i))
        if not has_non_matching[v]:
            leaf_nodes.append((v, i))

    def bfs(start):
        visited = [-1] * len(meta_adj)
        q = deque()
        q.append(start)
        visited[start] = 0
        max_dist = 0
        max_node = start
        while q:
            u = q.popleft()
            for v in meta_adj[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    q.append(v)
                    if visited[v] > max_dist:
                        max_dist = visited[v]
                        max_node = v
        return max_node, max_dist

    def find_diameter():
        u, _ = bfs(0)
        v, dist = bfs(u)
        return u, v, dist

    result = []
    remaining_meta = set(range(N//2))
    remaining_leaf_nodes = leaf_nodes.copy()

    while remaining_meta:
        if len(remaining_meta) == 1:
            m = remaining_meta.pop()
            u, v = original_pairs[m]
            result.append((u, v))
            break
        start = next(iter(remaining_meta))
        u, _ = bfs(start)
        v, diameter = bfs(u)
        m1 = u
        m2 = v

        m1_nodes = []
        for node, pair in remaining_leaf_nodes:
            if pair == m1:
                m1_nodes.append(node)
        m1_leaf = max(m1_nodes) if m1_nodes else original_pairs[m1][0]

        m2_nodes = []
        for node, pair in remaining_leaf_nodes:
            if pair == m2:
                m2_nodes.append(node)
        m2_leaf = max(m2_nodes) if m2_nodes else original_pairs[m2][0]

        result.append((m1_leaf, m2_leaf))

        remaining_meta.discard(m1)
        remaining_meta.discard(m2)
        remaining_leaf_nodes = [(node, pair) for node, pair in remaining_leaf_nodes if pair != m1 and pair != m2]

    print('
'.join(f"{a} {b}" for a, b in result))

if __name__ == '__main__':
    main()