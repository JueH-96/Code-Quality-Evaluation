import sys
import collections

def solve():
    N, M = map(int, sys.stdin.readline().split())

    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v, z = map(int, sys.stdin.readline().split())
        u -= 1 # 0-indexed
        v -= 1 # 0-indexed
        adj[u].append((v, z))
        adj[v].append((u, z))

    D_values = [-1] * N 
    visited = [False] * N
    final_A = [0] * N
    
    K_BITS = 30 # Max bit position needed for values up to 10^9 (2^29 < 10^9 < 2^30)

    for i in range(N):
        if not visited[i]:
            component_nodes = []
            
            D_values[i] = 0 
            visited[i] = True
            component_nodes.append(i)
            
            bfs_q_ptr = 0 
            possible_component = True
            
            while bfs_q_ptr < len(component_nodes):
                u = component_nodes[bfs_q_ptr]
                bfs_q_ptr += 1
                
                for v, z_uv in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        D_values[v] = D_values[u] ^ z_uv
                        component_nodes.append(v)
                    else:
                        if (D_values[u] ^ D_values[v]) != z_uv:
                            possible_component = False
                            break 
                if not possible_component:
                    break
            
            if not possible_component:
                print("-1")
                return

            xc = 0
            for bit_k in range(K_BITS - 1, -1, -1):
                cnt_D_bit_is_0 = 0 
                cnt_D_bit_is_1 = 0
                for node_idx in component_nodes:
                    if (D_values[node_idx] >> bit_k) & 1:
                        cnt_D_bit_is_1 += 1
                    else:
                        cnt_D_bit_is_0 += 1
                
                # If k-th bit of xc (b_k) is 0, sum of (b_k ^ D_node,k) is cnt_D_bit_is_1.
                # If b_k is 1, sum of (b_k ^ D_node,k) is cnt_D_bit_is_0.
                # Choose b_k to minimize this sum.
                # Set b_k=1 if cnt_D_bit_is_0 < cnt_D_bit_is_1. Otherwise b_k=0.
                if cnt_D_bit_is_0 < cnt_D_bit_is_1:
                    xc |= (1 << bit_k)
            
            for node_idx in component_nodes:
                final_A[node_idx] = xc ^ D_values[node_idx]
                
    print(*(final_A))

solve()