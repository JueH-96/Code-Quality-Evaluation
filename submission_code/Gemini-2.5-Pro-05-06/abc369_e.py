import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())

    all_bridges_data = [None] * (M + 1) # 1-indexed bridge storage
    
    sp_matrix = [[float('inf')] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        sp_matrix[i][i] = 0

    for i in range(1, M + 1):
        u, v, t = map(int, sys.stdin.readline().split())
        all_bridges_data[i] = {'U': u, 'V': v, 'T': t}
        sp_matrix[u][v] = min(sp_matrix[u][v], t)
        sp_matrix[v][u] = min(sp_matrix[v][u], t)

    for k_fw in range(1, N + 1):
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if sp_matrix[i][k_fw] != float('inf') and sp_matrix[k_fw][j] != float('inf'):
                    sp_matrix[i][j] = min(sp_matrix[i][j], sp_matrix[i][k_fw] + sp_matrix[k_fw][j])
    
    Q_count = int(sys.stdin.readline())
    results_output = []

    for _ in range(Q_count):
        query_line = list(map(int, sys.stdin.readline().split()))
        K = query_line[0]
        K_bridges_indices_input = query_line[1:]

        selected_bridges_info = []
        for bridge_idx in K_bridges_indices_input:
            selected_bridges_info.append(all_bridges_data[bridge_idx])

        # endpoints[k_s_idx][0] is U-node, endpoints[k_s_idx][1] is V-node
        # for the k_s_idx-th bridge in selected_bridges_info (0 to K-1)
        bridge_U_nodes = [selected_bridges_info[i]['U'] for i in range(K)]
        bridge_V_nodes = [selected_bridges_info[i]['V'] for i in range(K)]
        bridge_T_values = [selected_bridges_info[i]['T'] for i in range(K)]

        # DP state: dp[mask][last_k_s_idx][exit_d]
        # exit_d: 0 if exited from U-node of bridge, 1 if exited from V-node of bridge
        dp = [[[float('inf')] * 2 for _ in range(K)] for _ in range(1 << K)]

        # Base cases (mask has one bit set)
        for k_s_idx in range(K):
            mask = 1 << k_s_idx
            
            # Path: island 1 --SP--> U_k --bridge_k--> V_k. Exit at V_k (exit_d = 1)
            cost_to_exit_V = sp_matrix[1][bridge_U_nodes[k_s_idx]] + bridge_T_values[k_s_idx]
            dp[mask][k_s_idx][1] = cost_to_exit_V

            # Path: island 1 --SP--> V_k --bridge_k--> U_k. Exit at U_k (exit_d = 0)
            cost_to_exit_U = sp_matrix[1][bridge_V_nodes[k_s_idx]] + bridge_T_values[k_s_idx]
            dp[mask][k_s_idx][0] = cost_to_exit_U
        
        # DP iterations
        for i_mask in range(1, 1 << K):
            for curr_k_s_idx in range(K):
                if not (i_mask & (1 << curr_k_s_idx)):
                    continue
                
                prev_mask = i_mask ^ (1 << curr_k_s_idx)
                if prev_mask == 0: 
                    continue

                # Cost to reach curr_k_s_idx, exiting at its V-node (traverse U_curr -> V_curr)
                entry_node_for_curr_U = bridge_U_nodes[curr_k_s_idx]
                min_cost_to_exit_V = float('inf')
                for prev_k_s_idx in range(K):
                    if not (prev_mask & (1 << prev_k_s_idx)):
                        continue
                    
                    if dp[prev_mask][prev_k_s_idx][0] != float('inf'): # Prev exited U_prev
                        prev_exit_node = bridge_U_nodes[prev_k_s_idx]
                        cost = dp[prev_mask][prev_k_s_idx][0] + sp_matrix[prev_exit_node][entry_node_for_curr_U] + bridge_T_values[curr_k_s_idx]
                        min_cost_to_exit_V = min(min_cost_to_exit_V, cost)
                    
                    if dp[prev_mask][prev_k_s_idx][1] != float('inf'): # Prev exited V_prev
                        prev_exit_node = bridge_V_nodes[prev_k_s_idx]
                        cost = dp[prev_mask][prev_k_s_idx][1] + sp_matrix[prev_exit_node][entry_node_for_curr_U] + bridge_T_values[curr_k_s_idx]
                        min_cost_to_exit_V = min(min_cost_to_exit_V, cost)
                dp[i_mask][curr_k_s_idx][1] = min_cost_to_exit_V

                # Cost to reach curr_k_s_idx, exiting at its U-node (traverse V_curr -> U_curr)
                entry_node_for_curr_V = bridge_V_nodes[curr_k_s_idx]
                min_cost_to_exit_U = float('inf')
                for prev_k_s_idx in range(K):
                    if not (prev_mask & (1 << prev_k_s_idx)):
                        continue
                        
                    if dp[prev_mask][prev_k_s_idx][0] != float('inf'): # Prev exited U_prev
                        prev_exit_node = bridge_U_nodes[prev_k_s_idx]
                        cost = dp[prev_mask][prev_k_s_idx][0] + sp_matrix[prev_exit_node][entry_node_for_curr_V] + bridge_T_values[curr_k_s_idx]
                        min_cost_to_exit_U = min(min_cost_to_exit_U, cost)

                    if dp[prev_mask][prev_k_s_idx][1] != float('inf'): # Prev exited V_prev
                        prev_exit_node = bridge_V_nodes[prev_k_s_idx]
                        cost = dp[prev_mask][prev_k_s_idx][1] + sp_matrix[prev_exit_node][entry_node_for_curr_V] + bridge_T_values[curr_k_s_idx]
                        min_cost_to_exit_U = min(min_cost_to_exit_U, cost)
                dp[i_mask][curr_k_s_idx][0] = min_cost_to_exit_U
        
        min_total_travel_time = float('inf')
        final_mask = (1 << K) - 1
        for k_s_idx in range(K):
            # Last bridge was k_s_idx, exited at its U-node
            if dp[final_mask][k_s_idx][0] != float('inf'):
                last_node = bridge_U_nodes[k_s_idx]
                total_cost = dp[final_mask][k_s_idx][0] + sp_matrix[last_node][N]
                min_total_travel_time = min(min_total_travel_time, total_cost)

            # Last bridge was k_s_idx, exited at its V-node
            if dp[final_mask][k_s_idx][1] != float('inf'):
                last_node = bridge_V_nodes[k_s_idx]
                total_cost = dp[final_mask][k_s_idx][1] + sp_matrix[last_node][N]
                min_total_travel_time = min(min_total_travel_time, total_cost)
        
        results_output.append(str(min_total_travel_time))

    sys.stdout.write("
".join(results_output) + "
")

solve()