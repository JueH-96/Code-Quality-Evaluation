def solve():
    n = int(input())
    adj = [[] for _ in range(n)]
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))

    def get_snowflake_size(center, blue_nodes, adj):
        if not blue_nodes:
            return 0

        y = -1
        all_leaves = set()

        for blue_node in blue_nodes:
            current_leaves = set()
            for neighbor in adj[blue_node]:
                if neighbor != center:
                    is_leaf = True
                    for other_neighbor in adj[neighbor]:
                        if other_neighbor != blue_node:
                            is_leaf = False
                            break
                    if is_leaf:
                        current_leaves.add(neighbor)

            if y == -1:
                y = len(current_leaves)
            elif len(current_leaves) != y:
                return 0

            for leaf in current_leaves:
                if leaf in all_leaves:
                    return 0
                all_leaves.add(leaf)

        return 1 + len(blue_nodes) + len(blue_nodes) * y

    max_snowflake_vertices = 0
    for i in range(1 << n):
        remaining_nodes_indices = [j for j in range(n) if (i >> j) & 1]
        if not remaining_nodes_indices:
            continue

        sub_adj = [[] for _ in range(len(remaining_nodes_indices))]
        node_map = {original_index: new_index for new_index, original_index in enumerate(remaining_nodes_indices)}

        num_edges = 0
        is_tree = True

        if len(remaining_nodes_indices) > 1:
            for u_orig, v_orig in edges:
                if u_orig in node_map and v_orig in node_map:
                    u_new = node_map[u_orig]
                    v_new = node_map[v_orig]
                    sub_adj[u_new].append(v_new)
                    sub_adj[v_new].append(u_new)
                    num_edges += 1

            if num_edges != len(remaining_nodes_indices) - 1:
                is_tree = False
            else:
                if remaining_nodes_indices:
                    start_node = node_map[remaining_nodes_indices[0]]
                    visited = [False] * len(remaining_nodes_indices)
                    stack = [start_node]
                    visited_count = 0
                    while stack:
                        curr = stack.pop()
                        if not visited[curr]:
                            visited[curr] = True
                            visited_count += 1
                            for neighbor in sub_adj[curr]:
                                stack.append(neighbor)
                    if visited_count != len(remaining_nodes_indices):
                        is_tree = False

        if is_tree:
            if len(remaining_nodes_indices) >= 3:
                for center_orig in remaining_nodes_indices:
                    center_new = node_map[center_orig]
                    neighbors_orig = [node for node in adj[center_orig] if node in node_map]
                    neighbors_new_indices = [node_map[node] for node in neighbors_orig]

                    for x in range(1, len(neighbors_new_indices) + 1):
                        from itertools import combinations
                        for comb_indices in combinations(range(len(neighbors_new_indices)), x):
                            blue_nodes_orig = [remaining_nodes_indices[neighbors_new_indices[i]] for i in comb_indices]

                            current_y = -1
                            is_valid_snowflake = True
                            potential_leaves = set()

                            for blue_orig in blue_nodes_orig:
                                leaves_orig = set()
                                for leaf_candidate_orig in adj[blue_orig]:
                                    if leaf_candidate_orig != center_orig and leaf_candidate_orig in node_map:
                                        is_leaf = True
                                        for other_neighbor_orig in adj[leaf_candidate_orig]:
                                            if other_neighbor_orig != blue_orig and other_neighbor_orig in node_map:
                                                is_leaf = False
                                                break
                                        if is_leaf:
                                            leaves_orig.add(leaf_candidate_orig)

                                if current_y == -1:
                                    current_y = len(leaves_orig)
                                elif len(leaves_orig) != current_y:
                                    is_valid_snowflake = False
                                    break
                                for leaf in leaves_orig:
                                    if leaf in potential_leaves:
                                        is_valid_snowflake = False
                                        break
                                    potential_leaves.add(leaf)
                                if not is_valid_snowflake:
                                    break
                            if is_valid_snowflake and current_y >= 0:
                                max_snowflake_vertices = max(max_snowflake_vertices, len(remaining_nodes_indices))
            elif len(remaining_nodes_indices) > 0:
                max_snowflake_vertices = max(max_snowflake_vertices, len(remaining_nodes_indices))

    print(n - max_snowflake_vertices)

solve()