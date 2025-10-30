import sys

# Set recursion limit to accommodate deep Tries.
# Maximum possible depth of the Trie is the maximum string length, which can be up to total_length (3*10^5).
sys.setrecursionlimit(3 * 10**5 + 100) # Add a small buffer just in case

class TrieNode:
    """
    Represents a node in the Trie.
    """
    def __init__(self):
        self.children = {}  # Dictionary mapping characters to child TrieNode objects
        self.count = 0      # Number of strings that pass through this node
                            # (i.e., have the prefix represented by this node)

def solve():
    N = int(sys.stdin.readline())
    strings = []
    for _ in range(N):
        strings.append(sys.stdin.readline().strip())

    root = TrieNode()

    # Build the Trie and update counts for each node.
    # The `count` for a node `u` stores how many of the input strings have the prefix
    # represented by `u`.
    for s in strings:
        current_node = root
        for char in s:
            # If the child node for 'char' doesn't exist, create it.
            if char not in current_node.children:
                current_node.children[char] = TrieNode()
            
            # Move to the child node.
            current_node = current_node.children[char]
            
            # Increment the count for this node.
            # This correctly updates the number of strings passing through this prefix.
            current_node.count += 1 

    # Calculate the total sum of LCPs.
    # The total sum is equivalent to sum_{u in Trie} (count[u] * (count[u] - 1) / 2).
    # Each term (count[u] * (count[u] - 1) / 2) represents the number of pairs of strings
    # that share the prefix represented by node `u`.
    # Summing this over all nodes correctly accumulates the LCP contribution for all pairs.
    # For a pair (Si, Sj) with LCP length L, they will pass through L common nodes.
    # Each of these L common nodes will contribute 1 to the pair's total LCP.
    
    total_lcp_sum = 0

    # Use a recursive DFS to traverse the Trie and sum up contributions from all nodes.
    def calculate_sum_dfs(node):
        nonlocal total_lcp_sum
        
        # Add the contribution of the current node to the total sum.
        # If node.count < 2, the term `node.count * (node.count - 1) // 2` will be 0,
        # which is correct as no pairs can be formed.
        total_lcp_sum += node.count * (node.count - 1) // 2
        
        # Recursively call DFS for all child nodes.
        for child_node in node.children.values():
            calculate_sum_dfs(child_node)

    # Start the DFS traversal from the root.
    # The root node's count is 0 (as no strings technically pass through an "empty" prefix that is incremented).
    # So, its contribution to the sum will correctly be 0.
    calculate_sum_dfs(root) 

    # Print the final calculated sum.
    sys.stdout.write(str(total_lcp_sum) + "
")

# Call the solve function to run the program.
solve()