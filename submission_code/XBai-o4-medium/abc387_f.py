import sys
from collections import defaultdict

MOD = 998244353

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    A = [0] + A  # 1-based indexing

    # Step 1: Find cycle_mask
    def find_cycles(n, A):
        color = [0] * (n + 1)  # 0: unvisited, 1: visiting, 2: visited
        cycle_mask = [False] * (n + 1)
        for start in range(1, n + 1):
            if color[start] == 0:
                path = []
                current = start
                while True:
                    color[current] = 1
                    path.append(current)
                    next_node = A[current]
                    if color[next_node] == 0:
                        current = next_node
                    elif color[next_node] == 1:
                        # Found a cycle
                        idx = path.index(next_node)
                        cycle = path[idx:]
                        for node in cycle:
                            cycle_mask[node] = True
                        # Mark all in path as visited
                        for node in path:
                            color[node] = 2
                        break
                    else:
                        # color is 2
                        for node in path:
                            color[node] = 2
                        break
        return cycle_mask

    cycle_mask = find_cycles(N, A)

    # Step 2: Compute roots for non-cycle nodes
    roots = [0] * (N + 1)
    for i in range(1, N + 1):
        if not cycle_mask[i]:
            current = i
            while not cycle_mask[current]:
                current = A[current]
            roots[i] = current
        else:
            roots[i] = i

    # Step 3: Find all cycles (list of lists)
    processed_cycle = [False] * (N + 1)
    all_cycles = []
    for i in range(1, N + 1):
        if cycle_mask[i] and not processed_cycle[i]:
            # Find the entire cycle starting at i
            cycle = []
            current = i
            while True:
                cycle.append(current)
                processed_cycle[current] = True
                current = A[current]
                if current == i:
                    break
            all_cycles.append(cycle)

    # Step 4: For each cycle, process trees attached to it
    total = 1

    for cycle in all_cycles:
        # Collect all trees attached to this cycle
        trees_root_dp = []
        for u in cycle:
            # Collect all nodes with root u and not in cycle
            tree_nodes = []
            for node in range(1, N + 1):
                if roots[node] == u and not cycle_mask[node]:
                    tree_nodes.append(node)
            if not tree_nodes:
                continue
            # Build children dict for these tree_nodes
            children_dict = defaultdict(list)
            for node in tree_nodes:
                parent = A[node]
                children_dict[parent].append(node)
            # Process the tree with root u
            def process_tree(root, children_dict, mod, M_val):
                stack = [(root, False)]
                post_order = []
                while stack:
                    node, visited = stack.pop()
                    if not visited:
                        stack.append((node, True))
                        # Push children in reverse order to maintain order
                        for child in reversed(children_dict.get(node, [])):
                            stack.append((child, False))
                    else:
                        post_order.append(node)
                # Now compute dp and prefix for each node in post_order
                node_dp = {}  # key: node, value: (dp, prefix)
                for node in post_order:
                    ch = children_dict.get(node, [])
                    if not ch:
                        # Leaf node
                        dp_arr = [0] * (M_val + 1)
                        for k in range(1, M_val + 1):
                            dp_arr[k] = 1
                        prefix_arr = [0] * (M_val + 1)
                        for k in range(1, M_val + 1):
                            prefix_arr[k] = (prefix_arr[k - 1] + dp_arr[k]) % mod
                        node_dp[node] = (dp_arr, prefix_arr)
                    else:
                        # Compute product of children's prefix
                        product_arr = [1] * (M_val + 1)
                        for child in ch:
                            child_prefix = node_dp[child][1]
                            for k in range(1, M_val + 1):
                                product_arr[k] = product_arr[k] * child_prefix[k] % mod
                        # Compute prefix
                        prefix_arr = [0] * (M_val + 1)
                        for k in range(1, M_val + 1):
                            prefix_arr[k] = (prefix_arr[k - 1] + product_arr[k]) % mod
                        node_dp[node] = (product_arr, prefix_arr)
                return node_dp[root][0]

            # Call process_tree
            root_dp = process_tree(u, children_dict, MOD, M)
            trees_root_dp.append(root_dp)

        # Compute component contribution
        if not trees_root_dp:
            # Component contribution is M
            contrib = M % MOD
        else:
            contrib = 0
            for v in range(1, M + 1):
                product = 1
                for tree_dp in trees_root_dp:
                    product = product * tree_dp[v] % MOD
                contrib = (contrib + product) % MOD
        total = total * contrib % MOD

    print(total)

if __name__ == '__main__':
    main()