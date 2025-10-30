import sys
from collections import deque

def main():
    # Read input
    c = []
    for _ in range(3):
        row = list(map(int, sys.stdin.readline().split()))
        c.append(row)
    
    # Define cells as (i, j) where i and j are 1-based
    cells = [(i + 1, j + 1) for i in range(3) for j in range(3)]
    cell_to_idx = {cell: idx for idx, cell in enumerate(cells)}
    cell_values = {cell: c[i][j] for idx, (i, j) in enumerate(cells) for i in range(3) for j in range(3) if (i+1, j+1) == cell}
    
    # Define all lines
    lines = [
        [(1,1), (1,2), (1,3)],
        [(2,1), (2,2), (2,3)],
        [(3,1), (3,2), (3,3)],
        [(1,1), (2,1), (3,1)],
        [(1,2), (2,2), (3,2)],
        [(1,3), (2,3), (3,3)],
        [(1,1), (2,2), (3,3)],
        [(3,1), (2,2), (1,3)]
    ]
    
    S = []
    for line in lines:
        vals = [cell_values[cell] for cell in line]
        count = {}
        for v in vals:
            count[v] = count.get(v, 0) + 1
        if len(count) != 2:
            continue
        if 2 not in count.values():
            continue
        # Find Y and Xs
        Y_val = [v for v in count if count[v] == 1][0]
        X_val = [v for v in count if count[v] == 2][0]
        X_cells = [cell for cell in line if cell_values[cell] == X_val]
        Y_cell = [cell for cell in line if cell_values[cell] == Y_val][0]
        X1 = cell_to_idx[X_cells[0]]
        X2 = cell_to_idx[X_cells[1]]
        Y = cell_to_idx[Y_cell]
        S.append((X1, X2, Y))
    
    total = 0
    n = len(S)
    for mask in range(0, 1 << n):
        T = [S[i] for i in range(n) if (mask >> i) & 1]
        edges = set()
        for line in T:
            X1, X2, Y = line
            edges.add((X1, Y))
            edges.add((X2, Y))
        # Build adjacency list and in-degree
        adj = {u: [] for u in range(9)}
        in_degree = {u: 0 for u in range(9)}
        for u, v in edges:
            adj[u].append(v)
            in_degree[v] += 1
        # Check for cycles using topological sort
        in_degree_copy = in_degree.copy()
        queue = deque()
        for u in range(9):
            if in_degree_copy[u] == 0:
                queue.append(u)
        topo_order = []
        while queue:
            u = queue.popleft()
            topo_order.append(u)
            for v in adj[u]:
                in_degree_copy[v] -= 1
                if in_degree_copy[v] == 0:
                    queue.append(v)
        if len(topo_order) != 9:
            continue  # DAG has a cycle
        # Build parents for each node
        parents = {u: [] for u in range(9)}
        for u, v in edges:
            parents[v].append(u)
        # Compute number of topological sorts using DP
        dp = [0] * (1 << 9)
        dp[0] = 1
        for mask_dp in range(1 << 9):
            if dp[mask_dp] == 0:
                continue
            for u in range(9):
                if not (mask_dp & (1 << u)):
                    # Check if all parents are in mask_dp
                    all_parents_in = True
                    for p in parents[u]:
                        if not (mask_dp & (1 << p)):
                            all_parents_in = False
                            break
                    if all_parents_in:
                        new_mask = mask_dp | (1 << u)
                        dp[new_mask] += dp[mask_dp]
        count = dp[(1 << 9) - 1]
        term = count
        k = len(T)
        term *= (-1) ** k
        total += term
    
    # Calculate probability
    factorial_9 = 362880
    probability = total / factorial_9
    print("{0:.20f}".format(probability))

if __name__ == '__main__':
    main()