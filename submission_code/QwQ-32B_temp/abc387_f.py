import sys
MOD = 998244353

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))  # A[0] is A_1

    is_cycle = [False] * (N + 1)
    component = [0] * (N + 1)
    visited = [False] * (N + 1)

    for node in range(1, N+1):
        if not visited[node]:
            path = []
            current = node
            while True:
                if visited[current]:
                    if current in path:
                        idx = path.index(current)
                        cycle = path[idx:]
                        for n in cycle:
                            is_cycle[n] = True
                        root = cycle[0]
                        for n in path:
                            component[n] = root
                            visited[n] = True
                        break
                    else:
                        root = component[current]
                        for n in path:
                            component[n] = root
                            visited[n] = True
                        break
                else:
                    visited[current] = True
                    path.append(current)
                    current = A[current-1]  # A is 0-based

    root_of = [0] * (N + 1)
    for node in range(1, N+1):
        if is_cycle[node]:
            root_of[node] = node
        else:
            current = node
            while not is_cycle[current]:
                current = A[current-1]
            root_of[node] = current

    from collections import defaultdict
    cycle_nodes_by_component = defaultdict(list)
    for node in range(1, N+1):
        if is_cycle[node]:
            root = component[node]
            cycle_nodes_by_component[root].append(node)

    total = 1
    processed_components = set()

    for root in component:
        if root == 0 or root in processed_components:
            continue
        processed_components.add(root)
        cycle_nodes = cycle_nodes_by_component.get(root, [])
        dp_for_cycle_nodes = []

        for r in cycle_nodes:
            nodes_in_subtree = [r]
            for node in range(1, N+1):
                if root_of[node] == r and not is_cycle[node]:
                    nodes_in_subtree.append(node)

            children = defaultdict(list)
            for node in nodes_in_subtree:
                if node == r:
                    parent = A[node-1]
                    if parent in nodes_in_subtree:
                        children[parent].append(node)
                else:
                    parent = A[node-1]
                    if parent == r:
                        children[r].append(node)
                    else:
                        if parent in nodes_in_subtree:
                            children[parent].append(node)

            stack = [(r, False)]
            post_order = []
            visited_subtree = [False] * (N + 1)
            while stack:
                node, processed = stack.pop()
                if processed:
                    post_order.append(node)
                    continue
                visited_subtree[node] = True
                stack.append((node, True))
                for child in reversed(children[node]):
                    if not visited_subtree[child]:
                        stack.append((child, False))

            dp = [ [0]*(M+1) for _ in range(N+1) ]
            prefix = [ [0]*(M+1) for _ in range(N+1) ]

            for node in post_order:
                child_list = children[node]
                if not child_list:
                    for k in range(1, M+1):
                        dp[node][k] = 1
                    prefix[node][0] = 0
                    for k in range(1, M+1):
                        prefix[node][k] = (prefix[node][k-1] + 1) % MOD
                else:
                    for k in range(1, M+1):
                        res = 1
                        for child in child_list:
                            res = res * prefix[child][k] % MOD
                        dp[node][k] = res
                    prefix[node][0] = 0
                    for k in range(1, M+1):
                        prefix[node][k] = (prefix[node][k-1] + dp[node][k]) % MOD

            dp_for_cycle_nodes.append(dp[r])

        component_contribution = 0
        for v in range(1, M+1):
            product = 1
            for dp_r in dp_for_cycle_nodes:
                product = product * dp_r[v] % MOD
            component_contribution = (component_contribution + product) % MOD
        total = total * component_contribution % MOD

    print(total % MOD)

if __name__ == "__main__":
    main()