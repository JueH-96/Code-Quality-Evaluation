# YOUR CODE HERE
import sys

# Use fast I/O
input = sys.stdin.readline
# Use sys.stdout.write for potentially faster output than print
write = sys.stdout.write

# Define INF (infinity) using a large integer
# The maximum possible cost is bounded, and intermediate values like |S_j| - 2*LCP can be negative.
# 10**18 is sufficiently large and avoids potential float precision issues.
INF = 10**18 

# Trie Node Class Definition
class TrieNode:
    # Use __slots__ for memory optimization, reducing overhead per node object.
    # This can be significant given the potential number of nodes (up to sum of |S_i|).
    __slots__ = ['children', 'minL', 'minSubtreeL'] 
    
    def __init__(self):
        # 'children' stores mapping from character to child TrieNode. Using dict is flexible.
        self.children = {} 
        # 'minL' stores the minimum length of a string ending exactly at this node. Initialize to INF.
        self.minL = INF 
        # 'minSubtreeL' stores the minimum length of a string ending anywhere in the subtree rooted at this node (inclusive). Initialize to INF.
        self.minSubtreeL = INF 

# Main solver function
def solve():
    N = int(input())
    
    # Initialize the Trie with a root node
    root = TrieNode()
    
    # List to store the computed minimum costs for each k
    results = []

    # Process each string S_k from k = 1 to N
    for k in range(1, N + 1):
        # Read the k-th string
        S = input().strip()
        Sk_len = len(S)

        # Initialize variables for the current string processing
        current_node = root
        # 'path_nodes' keeps track of nodes along S_k's path for efficient updates later
        path_nodes = [root] 
        
        # 'current_min_val' tracks the minimum value of |S_j| - 2*LCP(S_k, S_j) found so far for j < k
        current_min_val = INF 
        
        # Flag to check if S_k's path completely existed in the Trie before adding it
        path_fully_existed = True 

        # Traverse the Trie along S_k's path character by character
        for p in range(Sk_len):
            char = S[p]
            
            # Check Case 3: Target strings S_j whose path diverges from S_k's path at the current node (depth p)
            # These strings S_j share a prefix of length p with S_k. LCP(S_k, S_j) = p.
            # We need to find the minimum |S_j| among such strings.
            # Iterate through children of the current node.
            for child_char, child_node in current_node.children.items():
                # If the child corresponds to a different character than S_k's path character...
                if child_char != char: 
                    # ...and if there's any string ending in that child's subtree...
                    if child_node.minSubtreeL != INF:
                         # ...update current_min_val using the formula: min_len - 2 * LCP_length
                         # Here, min_len is child_node.minSubtreeL, LCP_length is p.
                         current_min_val = min(current_min_val, child_node.minSubtreeL - 2 * p)

            # Check if the path for S_k continues with the current character 'char'
            if char not in current_node.children:
                # If not, S_k's path needs extension. Create new nodes.
                path_fully_existed = False
                # Create nodes for the remaining characters of S_k
                for i in range(p, Sk_len):
                    new_node = TrieNode()
                    current_node.children[S[i]] = new_node
                    current_node = new_node
                    path_nodes.append(current_node)
                # Path construction finished, break the traversal loop
                break 

            # If path continues, move to the existing child node
            current_node = current_node.children[char]
            # Add the current node to the path list
            path_nodes.append(current_node)

            # Check Case 1: A previous string S_j is a prefix of S_k, ending at this node
            # The current node is at depth p+1.
            # If any string S_j ends exactly at this node (minL is finite)...
            if current_node.minL != INF:
                 # ...update current_min_val. |S_j| is current_node.minL, LCP length is p+1.
                 # Formula: minL - 2 * LCP_length = minL - 2 * (p+1)
                 current_min_val = min(current_min_val, current_node.minL - 2 * (p + 1))

        # After the loop, if path_fully_existed is true, S_k's entire path was already in the Trie.
        if path_fully_existed:
             # current_node is now the node corresponding to the full string S_k, at depth Sk_len.
             
             # Check Case 2: S_k is a prefix of some previous string S_j
             # These S_j continue paths from current_node's children.
             for child_char, child_node in current_node.children.items():
                  # If strings end in the subtree of a child...
                  if child_node.minSubtreeL != INF:
                       # ...update current_min_val. Min length |S_j| is child_node.minSubtreeL. LCP is Sk_len.
                       # Formula: minSubtreeL - 2 * LCP_length = minSubtreeL - 2 * Sk_len
                       current_min_val = min(current_min_val, child_node.minSubtreeL - 2 * Sk_len)

             # Check Case 1 again specifically for the final node of S_k.
             # This handles the scenario where S_k is identical to some S_j (j < k).
             if current_node.minL != INF:
                 # |S_j| = current_node.minL (should be Sk_len if S_k == S_j). LCP is Sk_len.
                 # Formula: minL - 2 * LCP_length = minL - 2 * Sk_len
                 current_min_val = min(current_min_val, current_node.minL - 2 * Sk_len)


        # Calculate the minimum cost for S_k.
        # Default cost is Sk_len (cost to make S_k empty).
        ans_k = Sk_len 
        
        # If we found any potential match S_j (current_min_val is not INF)...
        if current_min_val != INF:
            # Calculate the cost to transform S_k into the best matching S_j found.
            # Total cost = Sk_len + (|S_j| - 2*LCP) = Sk_len + current_min_val
            cost_to_match_prev = Sk_len + current_min_val
            # The final answer is the minimum of making empty string vs matching S_j
            ans_k = min(ans_k, cost_to_match_prev)
        
        # Store the computed minimum cost for k
        results.append(ans_k)

        # Update the Trie with information about S_k for future queries (k+1, k+2, ...)
        
        # Update minL for the node where S_k ends.
        # path_nodes[-1] gives the final node for S_k.
        end_node_Sk = path_nodes[-1]
        
        # Update minL only if Sk_len is strictly smaller than the current minL.
        # This ensures minL stores the minimum length if multiple strings end at the same node.
        # If Sk_len == end_node_Sk.minL, no update necessary. If Sk_len > end_node_Sk.minL, keep the existing smaller minL.
        if Sk_len < end_node_Sk.minL:
             end_node_Sk.minL = Sk_len
             # Note: If minL is updated, minSubtreeL might also need update starting from this node.

        # Update minSubtreeL for all nodes on S_k's path, propagating Sk_len upwards from end node to root.
        # The value potentially improving minSubtreeL is Sk_len itself.
        new_min_subtree_val = Sk_len 
        
        # Iterate path nodes from end back towards the root
        for node in reversed(path_nodes): 
            # If Sk_len is smaller than the current minSubtreeL of the node...
            if new_min_subtree_val < node.minSubtreeL:
                 # ...update the node's minSubtreeL.
                 node.minSubtreeL = new_min_subtree_val
            else:
                 # Optimization: If Sk_len does not improve minSubtreeL for this node,
                 # it cannot improve it for any ancestor node via this specific path update.
                 # We can stop propagating upwards early.
                 break 

    # After processing all N strings, print the stored results.
    # Use "
".join for efficient string concatenation and single write call.
    output = "
".join(map(str, results))
    write(output + "
")

# Execute the main solver function
solve()