import sys

class TrieNode:
    def __init__(self):
        self.children = {} # char_code (0-25) -> TrieNode
        self.is_end_of_string = False
        self.min_len_in_subtree = float('inf') # minimum length of any string S_j that passes through this node or ends here

def solve():
    N = int(sys.stdin.readline())
    strings = [sys.stdin.readline().strip() for _ in range(N)]

    trie = TrieNode()

    results = []

    for k in range(N):
        S_k = strings[k]
        len_S_k = len(S_k)
        
        current_min_cost = len_S_k # Initial cost: make S_k empty

        node_ptr = trie
        current_matched_depth = 0 # Depth of node_ptr in the trie (length of prefix it represents)
        
        # Store nodes on path for later update in insertion phase
        # This list will contain (TrieNode_object, its_depth)
        nodes_on_path_for_update = [(trie, 0)] # Start with the root node

        # Query phase: Iterate through S_k to find min cost with existing S_j (j < k)
        for i in range(len_S_k + 1): # Loop from i=0 up to len_S_k (inclusive)
                                     # The last iteration (i == len_S_k) handles S_k being a prefix of S_j
            
            # Case A: S_j (from previous) whose LCP with S_k is `current_matched_depth`.
            # These are S_j that pass through `node_ptr` but take a different branch from S_k[i].
            if i < len_S_k:
                char_for_Sk_code = ord(S_k[i]) - ord('a')
                for c_code in range(26):
                    if c_code != char_for_Sk_code and c_code in node_ptr.children:
                        diverging_node = node_ptr.children[c_code]
                        # For any S_j in `diverging_node`'s subtree, LCP(S_k, S_j) is `current_matched_depth`.
                        # Cost = len(S_k) + len(S_j) - 2 * LCP(S_k, S_j)
                        # `len(S_j)` is `diverging_node.min_len_in_subtree`.
                        current_min_cost = min(current_min_cost, len_S_k + diverging_node.min_len_in_subtree - 2 * current_matched_depth)
            else: # i == len_S_k. `node_ptr` is the node representing S_k (if its path exists).
                # Case C: S_k is a prefix of S_j.
                # LCP(S_k, S_j) = len(S_k).
                # Cost = len(S_k) + len(S_j) - 2 * len(S_k) = len(S_j) - len(S_k).
                # `min(len(S_j))` for those is `node_ptr.min_len_in_subtree`.
                current_min_cost = min(current_min_cost, node_ptr.min_len_in_subtree - len_S_k)

            # Case B: S_j is a prefix of S_k (S_j ends at `node_ptr`).
            # LCP(S_k, S_j) = `current_matched_depth`.
            # Cost = len(S_k) + current_matched_depth - 2 * current_matched_depth = len(S_k) - current_matched_depth.
            if node_ptr.is_end_of_string:
                current_min_cost = min(current_min_cost, len_S_k - current_matched_depth)
            
            # If we're at the end of S_k's processing, stop advancing `node_ptr`.
            if i == len_S_k:
                break # Exit the query loop

            # Advance `node_ptr` to the next character in S_k
            char_for_Sk_code = ord(S_k[i]) - ord('a')
            if char_for_Sk_code not in node_ptr.children:
                # Create node if it doesn't exist for S_k's path.
                # This node will be filled with `inf` for `min_len_in_subtree`.
                node_ptr.children[char_for_Sk_code] = TrieNode()
            node_ptr = node_ptr.children[char_for_Sk_code]
            current_matched_depth += 1
            nodes_on_path_for_update.append((node_ptr, current_matched_depth)) # Add node for later update
        
        results.append(current_min_cost)

        # Insertion Phase: Add S_k to Trie and update min_len_in_subtree values.
        
        # The last node in `nodes_on_path_for_update` is the one representing S_k.
        final_node_S_k, final_depth_S_k = nodes_on_path_for_update[-1]
        
        # Mark the end of S_k.
        final_node_S_k.is_end_of_string = True
        
        # Update min_len_in_subtree for nodes on S_k's path, starting from the end node.
        # This propagates the minimum length upwards correctly.
        for i in range(len(nodes_on_path_for_update) - 1, -1, -1):
            node, depth = nodes_on_path_for_update[i]
            # `len_S_k` itself passes through `node`, so it's a candidate for `node.min_len_in_subtree`.
            node.min_len_in_subtree = min(node.min_len_in_subtree, len_S_k)
            
            # Also, the minimum length of strings passing through this node
            # could be determined by strings in its child subtrees (not just on S_k's path).
            # This is implicitly handled because `child.min_len_in_subtree` is also updated
            # to be `min(itself, len_S_k)` if the child is on the path.
            # However, for a generic `u.min_len_in_subtree`, it should be `min(len(S_x) if S_x ends at u, min(child.min_len_in_subtree for all children))`.
            # The way it's done here is `min(current_val, len_S_k)` which applies to all nodes on the path of S_k.
            # This works because any string `S_x` that contributes to `node.min_len_in_subtree` must either be `S_k` itself,
            # or it must pass through one of `node`'s children. If it passes through a child on `S_k`'s path, it
            # will be covered by `len_S_k`. If it passes through an *other* child, that child's `min_len_in_subtree`
            # would already reflect `len(S_x)` and this is what is queried.

    for res in results:
        sys.stdout.write(str(res) + "
")

solve()