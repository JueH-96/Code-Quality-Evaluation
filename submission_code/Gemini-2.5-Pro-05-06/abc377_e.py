import sys

def main():
    N, K = map(int, sys.stdin.readline().split())
    P_input = list(map(int, sys.stdin.readline().split()))

    # Convert to 0-indexed: P_adj[i] is what i maps to.
    # P_input[j] is P_{j+1} (1-indexed value at (j+1)-th position).
    # So P_adj[j] = P_input[j] - 1.
    P_adj = [p_val - 1 for p_val in P_input]

    ans = [0] * N
    visited = [False] * N

    for i in range(N):
        if visited[i]:
            continue

        current_cycle = []
        curr = i
        while not visited[curr]:
            visited[curr] = True
            current_cycle.append(curr)
            curr = P_adj[curr]
        
        cycle_len = len(current_cycle)
        
        # We need to compute (P_adj)^(2^K).
        # For an element in a cycle of length L, this is (P_adj)^(2^K mod L).
        effective_exponent = pow(2, K, cycle_len)

        for k_idx in range(cycle_len):
            original_node_in_cycle = current_cycle[k_idx]
            final_image_of_node = current_cycle[(k_idx + effective_exponent) % cycle_len]
            ans[original_node_in_cycle] = final_image_of_node
            
    # Convert 0-indexed ans back to 1-indexed for printing
    output_values = [str(val + 1) for val in ans]
    sys.stdout.write(" ".join(output_values) + "
")

if __name__ == '__main__':
    main()