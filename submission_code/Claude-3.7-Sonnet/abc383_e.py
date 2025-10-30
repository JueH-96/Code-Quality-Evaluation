try:
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    import itertools

def main():
    N, M, K = map(int, input().strip().split())
    
    # Initialize adjacency matrix for the graph
    adj_matrix = [[float('inf')] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        adj_matrix[i][i] = 0
    
    # Read the edges
    for _ in range(M):
        u, v, w = map(int, input().strip().split())
        adj_matrix[u][v] = w
        adj_matrix[v][u] = w
    
    # Read the sequences A and B
    A = list(map(int, input().strip().split()))
    B = list(map(int, input().strip().split()))
    
    # Compute f(x, y) for all pairs using modified Floyd-Warshall
    for k in range(1, N + 1):
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                adj_matrix[i][j] = min(adj_matrix[i][j], max(adj_matrix[i][k], adj_matrix[k][j]))
    
    # Construct the cost matrix for bipartite matching
    cost_matrix = []
    for i in range(K):
        row = []
        for j in range(K):
            row.append(adj_matrix[A[i]][B[j]])
        cost_matrix.append(row)
    
    # Try to use scipy for efficient Hungarian algorithm, otherwise use brute force
    if SCIPY_AVAILABLE:
        cost_matrix_np = np.array(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
        min_cost = cost_matrix_np[row_ind, col_ind].sum()
    else:
        min_cost = float('inf')
        for perm in itertools.permutations(range(K)):
            cost = sum(cost_matrix[i][perm[i]] for i in range(K))
            min_cost = min(min_cost, cost)
    
    print(int(min_cost))

if __name__ == "__main__":
    main()