import itertools

def solve():
    n, m = map(int, input().split())
    bridges = []
    for _ in range(m):
        u, v, t = map(int, input().split())
        bridges.append((u, v, t))

    q = int(input())
    queries = []
    for _ in range(q):
        k = int(input())
        queries.append(list(map(int, input().split())))

    def get_all_pairs_shortest_paths(num_nodes, edges):
        dist = [[float('inf')] * num_nodes for _ in range(num_nodes)]
        for i in range(num_nodes):
            dist[i][i] = 0
        for u, v, t in edges:
            dist[u - 1][v - 1] = min(dist[u - 1][v - 1], t)
            dist[v - 1][u - 1] = min(dist[v - 1][u - 1], t)

        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        return dist

    adj_edges = []
    for i in range(m):
        u, v, t = bridges[i]
        adj_edges.append((u, v, t))

    all_pairs_dist = get_all_pairs_shortest_paths(n, adj_edges)

    for query_bridges_indices in queries:
        required_bridges = [bridges[i - 1] for i in query_bridges_indices]
        k = len(required_bridges)
        min_query_time = float('inf')

        for perm_indices in itertools.permutations(range(k)):
            permutation = [required_bridges[i] for i in perm_indices]
            dp = {}  # (bridge_index, current_node) -> min_time

            first_bridge = permutation[0]
            u1, v1, t1 = first_bridge

            dp[(0, u1)] = all_pairs_dist[0][v1 - 1] + t1
            dp[(0, v1)] = all_pairs_dist[0][u1 - 1] + t1

            for i in range(1, k):
                current_bridge = permutation[i]
                ui, vi, ti = current_bridge
                prev_bridge = permutation[i - 1]
                prev_ui, prev_vi, prev_ti = prev_bridge

                dp[(i, ui)] = float('inf')
                dp[(i, vi)] = float('inf')

                if (i - 1, prev_ui) in dp:
                    dp[(i, ui)] = min(dp[(i, ui)], dp[(i - 1, prev_ui)] + all_pairs_dist[prev_ui - 1][vi - 1] + ti)
                    dp[(i, vi)] = min(dp[(i, vi)], dp[(i - 1, prev_ui)] + all_pairs_dist[prev_ui - 1][ui - 1] + ti)

                if (i - 1, prev_vi) in dp:
                    dp[(i, ui)] = min(dp[(i, ui)], dp[(i - 1, prev_vi)] + all_pairs_dist[prev_vi - 1][vi - 1] + ti)
                    dp[(i, vi)] = min(dp[(i, vi)], dp[(i - 1, prev_vi)] + all_pairs_dist[prev_vi - 1][ui - 1] + ti)

            if k > 0:
                last_bridge = permutation[k - 1]
                uk, vk, tk = last_bridge
                time_to_n = float('inf')
                if (k - 1, uk) in dp:
                    time_to_n = min(time_to_n, dp[(k - 1, uk)] + all_pairs_dist[uk - 1][n - 1])
                if (k - 1, vk) in dp:
                    time_to_n = min(time_to_n, dp[(k - 1, vk)] + all_pairs_dist[vk - 1][n - 1])
                min_query_time = min(min_query_time, time_to_n)
            elif n == 1:
                min_query_time = 0

        print(min_query_time if min_query_time != float('inf') else 0)

solve()