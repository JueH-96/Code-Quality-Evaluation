import sys

def main():
    # Read N. N itself is not strictly needed if we just process the list of strings.
    _ = int(sys.stdin.readline()) 
    
    # Read strings: S_1 ... S_N on a single line, space-separated.
    strings = sys.stdin.readline().split()

    # Trie representation: list of dictionaries.
    # Each element (node) is a dictionary: {'children': {}, 'pass_count': 0}
    # 'children' is a dict mapping a character (string of length 1) to its child_node_idx (int).
    # 'pass_count' is an int: how many input strings pass through this node.
    
    # Node 0 is the root. It doesn't correspond to any prefix character.
    # Its pass_count is not used in the sum calculation.
    trie_nodes = [{'children': {}, 'pass_count': 0}] 

    for s_val in strings: # s_val is one of the input strings S_i
        curr_node_idx = 0  # Start traversal from the root (node 0)
        for char_val in s_val: # Iterate over characters of the string s_val
            
            children_map = trie_nodes[curr_node_idx]['children']
            if char_val not in children_map:
                # If this path (character) doesn't exist from current node, create a new child node.
                # The ID for the new node is the current size of trie_nodes list (before appending).
                new_node_idx = len(trie_nodes) 
                children_map[char_val] = new_node_idx
                # Add the new node to trie_nodes.
                trie_nodes.append({'children': {}, 'pass_count': 0}) 
            
            # Move to the child node.
            curr_node_idx = children_map[char_val]
            # Increment the pass_count for this node, as the current string 's_val' passes through it.
            trie_nodes[curr_node_idx]['pass_count'] += 1
            
    total_lcp_sum = 0
    # Iterate through all nodes in the trie, except for the root (node 0).
    # Nodes are indexed from 0 to len(trie_nodes) - 1.
    # Root is node 0. Nodes 1 and onwards represent actual prefixes.
    for node_idx in range(1, len(trie_nodes)):
        count = trie_nodes[node_idx]['pass_count']
        # For each node, if 'count' strings pass through it, these strings form C(count, 2) pairs.
        # Each such pair shares the prefix corresponding to this node. This means this specific 
        # prefix (or more accurately, the edge leading to this node) contributes 1 to the LCP 
        # length for each of these C(count, 2) pairs.
        # Summing C(count, 2) over all prefix-nodes gives the desired total sum of LCPs.
        total_lcp_sum += count * (count - 1) // 2
        
    print(total_lcp_sum)

if __name__ == '__main__':
    main()