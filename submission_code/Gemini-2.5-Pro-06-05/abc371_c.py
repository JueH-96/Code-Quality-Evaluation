# YOUR CODE HERE
import sys
from itertools import permutations

def solve():
    """
    Reads graph and cost data, then calculates the minimum cost to make
    graph G and H isomorphic by modifying edges in H.
    """
    # 1. Read N, the number of vertices.
    # We use 0-indexed vertices internally for convenience.
    try:
        N = int(sys.stdin.readline())
    except (IOError, ValueError):
        # Handles empty input
        return

    # 2. Build adjacency matrix for graph G.
    adj_G = [[0] * N for _ in range(N)]
    if N > 0:
        M_G = int(sys.stdin.readline())
        for _ in range(M_G):
            u, v = map(int, sys.stdin.readline().split())
            adj_G[u - 1][v - 1] = 1
            adj_G[v - 1][u - 1] = 1

    # 3. Build adjacency matrix for graph H.
    adj_H = [[0] * N for _ in range(N)]
    if N > 0:
        M_H = int(sys.stdin.readline())
        for _ in range(M_H):
            a, b = map(int, sys.stdin.readline().split())
            adj_H[a - 1][b - 1] = 1
            adj_H[b - 1][a - 1] = 1

    # 4. Build the cost matrix for flipping edges in H.
    costs = [[0] * N for _ in range(N)]
    # For N <= 1, this loop range is empty, which is correct.
    for i in range(N - 1):
        cost_line = list(map(int, sys.stdin.readline().split()))
        for k, cost in enumerate(cost_line):
            j = i + 1 + k
            costs[i][j] = cost
            costs[j][i] = cost

    # If N is 0 or 1, graphs are trivially isomorphic with 0 cost.
    if N <= 1:
        print(0)
        return

    min_total_cost = float('inf')
    
    # 5. Iterate through all possible vertex permutations.
    vertices = range(N)
    for p in permutations(vertices):
        # p is a tuple representing the permutation. p[i] is the vertex in H
        # that corresponds to vertex i in G.
        
        current_cost = 0
        # Check all unique pairs of vertices (i, j) in G.
        for i in range(N):
            for j in range(i + 1, N):
                # Desired edge status is from G.
                g_has_edge = adj_G[i][j]
                
                # Corresponding vertices in H under permutation p.
                h_v1, h_v2 = p[i], p[j]
                
                # Current edge status in H.
                h_has_edge = adj_H[h_v1][h_v2]

                # If status needs to be flipped, add the cost.
                if g_has_edge != h_has_edge:
                    current_cost += costs[h_v1][h_v2]
        
        min_total_cost = min(min_total_cost, current_cost)

    # 6. Print the minimum cost found.
    print(min_total_cost)

solve()