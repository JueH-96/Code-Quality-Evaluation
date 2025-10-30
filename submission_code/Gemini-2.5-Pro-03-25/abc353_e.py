# YOUR CODE HERE
import sys

# Increase recursion depth limit for deep tries.
# The maximum depth of the trie could be up to the total length of all strings L,
# which can be up to 3*10^5. Python's default recursion depth is often around 1000.
# We need to increase it to avoid RecursionError for inputs with long strings or deep trie structures.
# Set a value slightly larger than the maximum possible depth L.
try:
    # Set recursion depth to L + a small buffer. Max L is 3*10^5
    sys.setrecursionlimit(300010) 
except OverflowError:
     # On some systems or platforms (like certain online judges),
     # setting a very high recursion limit might not be allowed or might fail.
     # If setting the limit fails, this recursive solution might raise RecursionError 
     # for deep inputs. An iterative DFS implementation using an explicit stack 
     # would be an alternative to avoid recursion depth issues.
     pass 

class TrieNode:
    """Represents a node in the Trie data structure."""
    def __init__(self):
        # Children are stored in a dictionary where keys are characters
        # and values are child TrieNode objects.
        self.children = {} 
        # Stores the count of strings that end exactly at this node.
        self.end_count = 0
        # Optional attribute to store pass_count, primarily for clarity or debugging.
        # The value is computed and returned by the DFS function anyway.
        # self.pass_count = 0 

def solve():
    """
    Reads input strings, builds a Trie, performs a Depth First Search (DFS) 
    to calculate the sum of Longest Common Prefix (LCP) lengths for all pairs of strings,
    and prints the final result.
    """
    
    # Read the number of strings
    N = int(sys.stdin.readline())
    # Read the strings. Assumes they are provided on a single line, separated by spaces.
    strings = sys.stdin.readline().split()

    # Initialize the root node of the Trie
    root = TrieNode()
    
    # Build the Trie by inserting all N strings
    for S in strings:
        node = root
        for char in S:
            # Traverse down the trie. If a path for the character doesn't exist, create it.
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        # Increment the count of strings ending at the final node for string S.
        node.end_count += 1

    # Initialize the total sum of LCP lengths. This variable will be updated during DFS.
    total_lcp_sum = 0

    def dfs(node):
        """
        Performs Depth First Search starting from the given `node`.
        
        This function calculates the number of strings whose paths pass through `node` 
        (including strings that end exactly at `node`). It does this by summing the 
        `end_count` at the current node and the pass counts returned by recursive calls 
        to its children.
        
        Crucially, it updates the `total_lcp_sum`. For each child node, if `k` strings pass 
        through it, these `k` strings share the prefix corresponding to the child node. 
        The edge connecting the current `node` to its `child_node` represents adding one character
        to the prefix. This edge contributes to the LCP length of any pair among these `k` strings.
        The total contribution of this edge to the sum is the number of pairs, which is C(k, 2).
        By summing C(k, 2) for all nodes (implicitly, via their incoming edges), we get the total sum.

        Args:
            node: The current TrieNode being visited.

        Returns:
            int: The total number of strings passing through `node`.
        """
        # Use `nonlocal` to indicate that `total_lcp_sum` refers to the variable 
        # in the enclosing `solve` function scope, allowing modification.
        nonlocal total_lcp_sum 

        # Start the pass count for the current node with the number of strings ending exactly here.
        current_pass_count = node.end_count
        
        # Iterate through all children of the current node.
        # The `node.children.values()` provides direct access to child TrieNode objects.
        for child_node in node.children.values(): 
            # Recursively call DFS on the child node. The call returns the number of strings
            # passing through that child node.
            child_pass_count = dfs(child_node) 
            
            # Add the pass count from the child to the current node's pass count.
            # This aggregates counts up the tree.
            current_pass_count += child_pass_count
            
            # Calculate the contribution of the edge leading to this child node to the total LCP sum.
            # If `k = child_pass_count` strings pass through the child node, there are C(k, 2) pairs
            # among them. Each pair's LCP length includes the path up to the child node.
            # The formula C(k, 2) = k * (k-1) / 2 gives the number of pairs.
            # We only add to the sum if k >= 2, since C(k, 2) is 0 for k < 2.
            count = child_pass_count
            if count >= 2:
                # Use integer division // for C(k, 2) calculation.
                total_lcp_sum += count * (count - 1) // 2

        # Optionally store the computed pass count in the node object itself.
        # node.pass_count = current_pass_count 
        
        # Return the total pass count for this node. This value will be used by its parent node in the DFS.
        return current_pass_count

    # Initiate the DFS traversal starting from the root node.
    # The value returned by dfs(root) will be N (the total number of strings),
    # but we are primarily interested in the side effect: updating `total_lcp_sum`.
    dfs(root) 
    
    # After the DFS completes, `total_lcp_sum` holds the final answer. Print it to standard output.
    print(total_lcp_sum)

# Execute the main logic encapsulated in the solve function.
solve()