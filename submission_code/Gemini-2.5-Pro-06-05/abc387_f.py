import sys

# It's good practice to increase recursion limit for competitive programming,
# especially when using recursive approaches like DSU or DFS.
# Although this solution is mostly iterative, the DSU find method is recursive.
sys.setrecursionlimit(2 * 10**5)

MOD = 998244353

class DSU:
    """Disjoint Set Union data structure."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.num_elements = [1] * n

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Union by size
            if self.num_elements[root_i] < self.num_elements[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            self.num_elements[root_i] += self.num_elements[root_j]
            return True
        return False

def solve():
    """
    Main function to solve the problem.
    """
    try:
        input = sys.stdin.readline
        N, M = map(int, input().split())
        A = list(map(int, input().split()))
    except (IOError, ValueError):
        # Handle potential empty input on some platforms
        return

    A_adj = [x - 1 for x in A]

    dsu = DSU(N)
    for i in range(N):
        dsu.union(i, A_adj[i])

    components = {}
    for i in range(N):
        root = dsu.find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)

    total_ways = 1

    for comp_nodes in components.values():
        # Find cycle in the component
        u = comp_nodes[0]
        path = {}
        path_len = 0
        while u not in path:
            path[u] = path_len
            path_len += 1
            u = A_adj[u]
        
        cycle_start_node = u
        cycle_nodes = set()
        curr = cycle_start_node
        while True:
            cycle_nodes.add(curr)
            curr = A_adj[curr]
            if curr == cycle_start_node:
                break

        non_cycle_nodes = [node for node in comp_nodes if node not in cycle_nodes]
        
        if not non_cycle_nodes:
            total_ways = (total_ways * M) % MOD
            continue

        # DP on the forest of non-cycle nodes
        non_cycle_set = set(non_cycle_nodes)
        children = {node: [] for node in non_cycle_nodes}
        out_degree = {node: 0 for node in non_cycle_nodes}

        for node in non_cycle_nodes:
            parent = A_adj[node]
            if parent in non_cycle_set:
                children[parent].append(node)
                out_degree[parent] += 1
        
        queue = [node for node in non_cycle_nodes if out_degree[node] == 0]
        topo_order = []
        head = 0
        while head < len(queue):
            curr_node = queue[head]
            head += 1
            topo_order.append(curr_node)
            
            parent = A_adj[curr_node]
            if parent in non_cycle_set:
                out_degree[parent] -= 1
                if out_degree[parent] == 0:
                    queue.append(parent)
        
        F = {}
        
        for u_node in topo_order:
            F[u_node] = [0] * (M + 1)
            for k in range(1, M + 1):
                dp_val_k = 1
                for child in children[u_node]:
                    dp_val_k = (dp_val_k * F[child][k]) % MOD
                F[u_node][k] = (F[u_node][k-1] + dp_val_k) % MOD
        
        forest_roots = [node for node in non_cycle_nodes if A_adj[node] in cycle_nodes]
        
        product_poly = [1] * (M + 1)
        for root in forest_roots:
            for v in range(1, M + 1):
                product_poly[v] = (product_poly[v] * F[root][v]) % MOD
        
        comp_ways = 0
        for v in range(1, M + 1):
            comp_ways = (comp_ways + product_poly[v]) % MOD
            
        total_ways = (total_ways * comp_ways) % MOD

    print(total_ways)

if __name__ == "__main__":
    solve()