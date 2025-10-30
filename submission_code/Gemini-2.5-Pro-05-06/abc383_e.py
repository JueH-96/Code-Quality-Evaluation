import sys

def solve():
    N, M, K = map(int, sys.stdin.readline().split())

    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        edges.append((w, u, v))

    A_nodes_input = list(map(int, sys.stdin.readline().split()))
    B_nodes_input = list(map(int, sys.stdin.readline().split()))

    edges.sort()

    parent = list(range(N + 1))
    # sz for union by size optimization
    sz = [1] * (N + 1) 
    
    num_A = [0] * (N + 1)
    num_B = [0] * (N + 1)

    for node_vertex in A_nodes_input:
        num_A[node_vertex] += 1
    
    for node_vertex in B_nodes_input:
        num_B[node_vertex] += 1

    def find(i):
        # Path compression
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    total_cost = 0
    pairs_made = 0

    for w, u_node, v_node in edges:
        if pairs_made == K:
            break

        rootU = find(u_node)
        rootV = find(v_node)

        if rootU != rootV:
            cnt_A_U = num_A[rootU]
            cnt_B_U = num_B[rootU]
            cnt_A_V = num_A[rootV]
            cnt_B_V = num_B[rootV]

            pairs_already_possible_in_U = min(cnt_A_U, cnt_B_U)
            pairs_already_possible_in_V = min(cnt_A_V, cnt_B_V)
            
            total_A_in_merged = cnt_A_U + cnt_A_V
            total_B_in_merged = cnt_B_U + cnt_B_V
            
            pairs_possible_in_merged = min(total_A_in_merged, total_B_in_merged)
            
            newly_formable_pairs = pairs_possible_in_merged - pairs_already_possible_in_U - pairs_already_possible_in_V
            
            actual_match_count = min(newly_formable_pairs, K - pairs_made)
            
            if actual_match_count > 0:
                total_cost += actual_match_count * w
                pairs_made += actual_match_count
            
            # Union by size
            if sz[rootU] < sz[rootV]:
                parent[rootU] = rootV
                sz[rootV] += sz[rootU]
                num_A[rootV] = total_A_in_merged
                num_B[rootV] = total_B_in_merged
            else:
                parent[rootV] = rootU
                sz[rootU] += sz[rootV]
                num_A[rootU] = total_A_in_merged
                num_B[rootU] = total_B_in_merged
                
    sys.stdout.write(str(total_cost) + "
")

solve()