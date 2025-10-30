import collections

def solve():
    n = int(input())
    adj = collections.defaultdict(list)
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))

    def get_leaves(current_adj_nodes, current_adj):
        leaves = []
        for node in current_adj_nodes:
            if sum(1 for neighbor in current_adj.get(node, []) if neighbor in current_adj_nodes) == 1:
                leaves.append(node)
        if len(current_adj_nodes) == 1 and current_adj_nodes[0] not in current_adj:
            return current_adj_nodes
        elif len(current_adj_nodes) == 1 and len(current_adj.get(current_adj_nodes[0], [])) == 0:
            return current_adj_nodes
        elif len(current_adj_nodes) == 2 and any(v in current_adj.get(current_adj_nodes[0], []) for v in current_adj_nodes[1:]):
            return current_adj_nodes
        return leaves

    def distance(start_node, end_node, current_adj_nodes, current_adj):
        if start_node == end_node:
            return 0
        queue = collections.deque([(start_node, 0)])
        visited = {start_node}
        while queue:
            current_node, dist = queue.popleft()
            if current_node == end_node:
                return dist
            for neighbor in current_adj.get(current_node, []):
                if neighbor in current_adj_nodes and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return -1

    remaining_nodes = list(range(1, n + 1))
    results = []

    for _ in range(n // 2):
        current_leaves = get_leaves(remaining_nodes, adj)
        max_dist = -1
        best_u = -1
        best_v = -1

        for i in range(len(current_leaves)):
            for j in range(i + 1, len(current_leaves)):
                u, v = current_leaves[i], current_leaves[j]
                dist = distance(u, v, remaining_nodes, adj)
                if dist > max_dist:
                    max_dist = dist
                    best_u = u
                    best_v = v

        results.append((best_u, best_v))
        remaining_nodes.remove(best_u)
        remaining_nodes.remove(best_v)

    for u, v in results:
        print(u, v)

solve()