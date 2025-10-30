# YOUR CODE HERE
import heapq

def solve():
    n = int(input())
    edges = []
    adj = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v, l = map(int, input().split())
        u -= 1
        v -= 1
        edges.append((u, v, l))
        adj[u].append((v, l))
        adj[v].append((u, l))

    def get_dist(start_node, end_node):
        dist = [-1] * n
        dist[start_node] = 0
        pq = [(0, start_node)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == end_node:
                return d
            
            for v, l in adj[u]:
                if dist[v] == -1 or dist[v] > dist[u] + l:
                    dist[v] = dist[u] + l
                    heapq.heappush(pq, (dist[v], v))
        return -1

    results = []
    for k in range(1, n + 1):
        max_score = 0
        for i in range(1 << n):
            chosen_nodes = []
            for j in range(n):
                if (i >> j) & 1:
                    chosen_nodes.append(j)
            
            if len(chosen_nodes) != k:
                continue

            min_walk_len = float('inf')
            import itertools
            for path_nodes in itertools.permutations(chosen_nodes):
                walk_len = 0
                current_node = 0
                for next_node in path_nodes:
                    walk_len += get_dist(current_node, next_node)
                    current_node = next_node
                walk_len += get_dist(current_node, 0)
                min_walk_len = min(min_walk_len, walk_len)

            max_score = max(max_score, min_walk_len)
        results.append(max_score)

    for result in results:
        print(result)

solve()