import sys

class Node:
    def __init__(self):
        self.children = {}  # char -> node_idx
        self.is_end = False
        self.min_len_in_subtree = float('inf')

def solve():
    N = int(sys.stdin.readline())

    nodes = [Node()]  # Root node at index 0
    
    results = []

    for _idx_k in range(N):
        S_k = sys.stdin.readline().strip()
        len_S_k = len(S_k)

        M_k = -float('inf') 
        
        curr_node_idx = 0  # Start at root for traversal
        
        # Iterate over prefixes of S_k. i is length of current prefix S_k[0...i-1].
        # curr_node_idx is the trie node for this prefix.
        for i in range(len_S_k + 1):  # loop i from 0 to len_S_k
            u_node_obj = nodes[curr_node_idx]
            
            if u_node_obj.is_end:
                # The prefix S_k[0...i-1] (length i) is an existing string S_j.
                # LCP(S_k, S_j) = i, and |S_j| = i.
                # The term 2 * LCP(S_k, S_j) - |S_j| becomes 2*i - i = i.
                M_k = max(M_k, i)

            if i == len_S_k:
                # Current prefix is S_k itself. LCP is len_S_k = i.
                # Consider S_j where S_k is a prefix of S_j.
                # Such S_j paths go through a child of u_node_obj.
                # We want to maximize 2*i - |S_j|, so minimize |S_j|.
                # min |S_j| is nodes[child_idx].min_len_in_subtree.
                for child_char in u_node_obj.children:
                    child_idx = u_node_obj.children[child_char]
                    # Check if any string actually ends in this child's subtree
                    if nodes[child_idx].min_len_in_subtree != float('inf'):
                         M_k = max(M_k, 2 * i - nodes[child_idx].min_len_in_subtree)
                break  # All of S_k processed as a prefix, exit loop for i

            # Current prefix is S_k[0...i-1], next char in S_k is S_k[i].
            # Consider S_j that branch off at u_node_obj. LCP is i.
            # We want to maximize 2*i - |S_j|, so minimize |S_j|.
            # min |S_j| for strings in subtree of child is nodes[child_idx].min_len_in_subtree.
            for child_char in u_node_obj.children:
                child_idx = u_node_obj.children[child_char]
                if child_char != S_k[i]:  # This child's edge is NOT S_k[i]
                    if nodes[child_idx].min_len_in_subtree != float('inf'):
                        M_k = max(M_k, 2 * i - nodes[child_idx].min_len_in_subtree)
            
            # Advance along S_k's path in trie
            current_char_in_Sk = S_k[i]
            if current_char_in_Sk not in u_node_obj.children:
                # S_k's path diverges from trie. No S_j has S_k[0...i] as prefix.
                # All relevant LCPs with existing strings have been found.
                break  # Exit loop for i
            
            curr_node_idx = u_node_obj.children[current_char_in_Sk]
        
        # Calculate answer for S_k
        current_ans = len_S_k - max(0, M_k)
        results.append(str(current_ans))

        # Add S_k to trie and update min_len_in_subtree for nodes on its path
        curr_node_idx = 0  # Start at root for adding S_k
        path_nodes_indices_for_update = [0]  # Root is on path
        
        for char_idx_in_S_k in range(len_S_k):
            char_Sk = S_k[char_idx_in_S_k]
            u_node_obj = nodes[curr_node_idx]
            
            if char_Sk not in u_node_obj.children:
                new_node_idx = len(nodes)
                nodes.append(Node())
                u_node_obj.children[char_Sk] = new_node_idx
            
            curr_node_idx = u_node_obj.children[char_Sk]
            path_nodes_indices_for_update.append(curr_node_idx)
        
        nodes[curr_node_idx].is_end = True  # Mark end of S_k
        
        for node_idx_on_path in path_nodes_indices_for_update:
            nodes[node_idx_on_path].min_len_in_subtree = \
                min(nodes[node_idx_on_path].min_len_in_subtree, len_S_k)

    sys.stdout.write("
".join(results) + "
")

solve()