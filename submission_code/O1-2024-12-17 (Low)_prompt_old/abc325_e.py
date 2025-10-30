def solve():
    import sys
    import heapq
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = int(input_data[1])
    B = int(input_data[2])
    C = int(input_data[3])
    
    # Distances D[i][j] (0-based indexing internally)
    D = []
    idx = 4
    for i in range(N):
        row = list(map(int, input_data[idx:idx+N]))
        idx += N
        D.append(row)
    
    # We will build a graph with 2*N + 1 nodes:
    #   - Node 0 is a "virtual source"
    #   - Nodes 1..N correspond to (city i in "car" mode), i = 1..N
    #   - Nodes N+1..2N correspond to (city i in "train" mode), i = 1..N
    #
    # We'll run Dijkstra from node 0, and the answer will be min(dist[N], dist[2N]) 
    # which are the distances to city N in car-mode or train-mode.
    #
    # Edges:
    # - From node 0 to (1 in car-mode) and (1 in train-mode) with cost 0 
    #   (allowing us to start from city 1 by either car or train).
    # - Between city i and j in car-mode: cost = D[i-1][j-1] * A  (both directions)
    # - Between city i and j in train-mode: cost = D[i-1][j-1] * B + C (both directions)
    # - Switch from (i in car-mode) to (i in train-mode) with cost 0 (one-way).

    # For convenience in code:
    #   car_node(i) = i
    #   train_node(i) = i + N
    #   i is in [1..N]
    #   virtual_source = 0

    # Build adjacency list
    # Each entry of graph[u] is a list of (v, cost).
    graph = [[] for _ in range(2*N + 1)]
    
    # Function to add edges
    def add_edge(u, v, w):
        graph[u].append((v, w))
    
    # Virtual source edges (start from city 1 by either mode)
    #   0 -> 1   (car at city 1), cost=0
    #   0 -> N+1 (train at city 1), cost=0
    add_edge(0, 1, 0)
    add_edge(0, N+1, 0)
    
    # Add edges for the same mode
    # For car mode: i->j cost = D[i-1][j-1]*A, for i != j
    # We do both directions
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            dist_ij = D[i-1][j-1]
            cost_car = dist_ij * A
            # car(i) -> car(j)
            add_edge(i, j, cost_car)
            # car(j) -> car(i)
            add_edge(j, i, cost_car)
    
    # For train mode: (i+N)->(j+N) cost = D[i-1][j-1]*B + C, for i != j
    # We do both directions
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            dist_ij = D[i-1][j-1]
            cost_train = dist_ij * B + C
            add_edge(i+N, j+N, cost_train)
            add_edge(j+N, i+N, cost_train)
    
    # Add edges for switching from car to train
    for i in range(1, N+1):
        add_edge(i, i+N, 0)  # switch in city i
    
    # Now run Dijkstra from node 0
    INF = 10**18
    dist = [INF]*(2*N + 1)
    dist[0] = 0
    pq = [(0, 0)]  # (distance, node)
    
    while pq:
        cur_d, u = heapq.heappop(pq)
        if dist[u] < cur_d:
            continue
        for v, cost in graph[u]:
            nd = cur_d + cost
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    
    # Answer is min of dist[N] and dist[2N] (city N in car-mode or train-mode)
    ans = min(dist[N], dist[N+N])
    print(ans)

def main():
    solve()

if __name__ == "__main__":
    main()