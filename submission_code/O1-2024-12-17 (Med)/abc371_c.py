def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parsing the input
    # N
    idx = 0
    N = int(input_data[idx]); idx+=1

    # Graph G
    M_G = int(input_data[idx]); idx+=1
    g = [[0]*N for _ in range(N)]
    for _ in range(M_G):
        u = int(input_data[idx]); idx+=1
        v = int(input_data[idx]); idx+=1
        g[u-1][v-1] = 1
        g[v-1][u-1] = 1

    # Graph H
    M_H = int(input_data[idx]); idx+=1
    h = [[0]*N for _ in range(N)]
    for _ in range(M_H):
        a = int(input_data[idx]); idx+=1
        b = int(input_data[idx]); idx+=1
        h[a-1][b-1] = 1
        h[b-1][a-1] = 1

    # Cost matrix A (store symmetrically)
    cost = [[0]*N for _ in range(N)]
    for i in range(N-1):
        # read N - 1 - i numbers
        for j in range(i+1, N):
            A_ij = int(input_data[idx])
            idx+=1
            cost[i][j] = A_ij
            cost[j][i] = A_ij

    # Edge case: if N == 1, cost is always 0
    if N == 1:
        print(0)
        return

    from itertools import permutations

    INF = 10**15
    best_cost = INF

    # Try all permutations of vertices [0..N-1]
    # p[i] = which vertex of H corresponds to vertex i of G
    # We want adjacency of G[i][j] == adjacency of H[p[i]][p[j]]
    # If not, pay cost[p[i]][p[j]] to flip that edge in H.
    for p in permutations(range(N)):
        current_cost = 0
        for i in range(N):
            for j in range(i+1, N):
                if g[i][j] != h[p[i]][p[j]]:
                    current_cost += cost[p[i]][p[j]]
                    if current_cost > best_cost:
                        break
            if current_cost > best_cost:
                break
        if current_cost < best_cost:
            best_cost = current_cost

    print(best_cost)

# Do not forget to call main().
if __name__ == "__main__":
    main()