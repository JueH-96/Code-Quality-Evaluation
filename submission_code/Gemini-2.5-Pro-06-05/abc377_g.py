import sys

def solve():
    """
    Main function to solve the problem.
    Reads input, runs the algorithm, and prints output.
    """
    # Use fast I/O
    input = sys.stdin.readline

    try:
        N = int(input())
        if N == 0:
            return
    except (IOError, ValueError):
        # Handle empty input or malformed N
        return
        
    # Trie implementation
    # Using a list of dicts to represent nodes
    # trie[i] is the i-th node
    # Each node is a dict: {'children': {}, 'is_end': False, 'min_len': float('inf')}
    trie = [{'children': {}, 'is_end': False, 'min_len': float('inf')}]
    
    for _ in range(N):
        Sk = input().strip()
        len_Sk = len(Sk)
        
        # --- Query Phase ---
        # Find min_{j<k} { |S_j| + |S_k| - 2 * lcp(S_k, S_j) }
        # This is equivalent to finding |S_k| + min_{j<k} { |S_j| - 2 * lcp(S_k, S_j) }
        
        min_val = float('inf')
        
        node_idx = 0
        lcp_len = 0 # current prefix length
        
        for char_k in Sk:
            if node_idx == -1:
                break
            
            node = trie[node_idx]
            
            # Consider S_j where lcp(S_k, S_j) = lcp_len
            
            # Case 1: S_j is the current prefix S_k[0...lcp_len-1]
            if node['is_end']:
                min_val = min(min_val, lcp_len - 2 * lcp_len)

            # Case 2: S_j's path diverges from S_k's path at this node
            for child_char, child_idx in node['children'].items():
                if child_char != char_k:
                    child_node = trie[child_idx]
                    # min_len in the child's subtree gives min len(S_j) for this branch
                    min_val = min(min_val, child_node['min_len'] - 2 * lcp_len)

            if char_k not in node['children']:
                node_idx = -1 # Path for S_k does not continue in the trie
            else:
                node_idx = node['children'][char_k]
                lcp_len += 1

        # After the loop, if node_idx is valid, S_k's path is fully in the trie.
        # This means S_k is a prefix of (or equal to) some S_j.
        # lcp(S_k, S_j) will be len_Sk for all such S_j.
        if node_idx != -1:
            node = trie[node_idx]
            # min_len in this node's subtree gives min len(S_j) for which S_k is a prefix.
            if node['min_len'] != float('inf'):
                 min_val = min(min_val, node['min_len'] - 2 * len_Sk)

        # The cost to transform S_k to the best S_j is |S_k| + min_val
        cost_transform = float('inf')
        if min_val != float('inf'):
            cost_transform = len_Sk + min_val

        # The cost to make S_k empty is |S_k|
        cost_empty = len_Sk

        ans = min(cost_empty, cost_transform)
        sys.stdout.write(str(ans) + '
')

        # --- Update Phase ---
        # Insert Sk into the trie
        node_idx = 0
        trie[node_idx]['min_len'] = min(trie[node_idx]['min_len'], len_Sk)
        
        for char_k in Sk:
            if char_k not in trie[node_idx]['children']:
                trie.append({'children': {}, 'is_end': False, 'min_len': float('inf')})
                new_node_idx = len(trie) - 1
                trie[node_idx]['children'][char_k] = new_node_idx
            
            node_idx = trie[node_idx]['children'][char_k]
            trie[node_idx]['min_len'] = min(trie[node_idx]['min_len'], len_Sk)
            
        trie[node_idx]['is_end'] = True

if __name__ == "__main__":
    solve()