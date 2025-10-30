import sys

# Increase recursion depth
# The maximum depth of recursive calls is the height of the trie.
# The height of the trie is at most the maximum length of a string.
# The maximum total length of all strings is 3 * 10^5.
# The number of nodes in the trie is at most 1 + sum(|S_i|), which is at most 300001.
# A safe recursion depth limit is slightly more than the maximum possible number of nodes.
# The actual value is set inside the solve function after building the trie.


class TrieNode:
    def __init__(self):
        # children: dictionary mapping character to node index in trie_nodes list
        self.children = {}
        # end_of_string_count: number of strings ending exactly at this node
        self.end_of_string_count = 0
        # descendant_count: number of original strings whose path goes through this node
        # This will be computed in a post-order traversal after building the trie.
        self.descendant_count = 0

# Use a list as the storage for nodes. Index 0 is the root.
trie_nodes = []

# Insert a string into the trie
def insert(s):
    # Start at the root node (index 0)
    node_idx = 0
    for char in s:
        # If the current node does not have a child edge for the character, create a new node
        if char not in trie_nodes[node_idx].children:
            new_node_idx = len(trie_nodes)
            trie_nodes.append(TrieNode())
            trie_nodes[node_idx].children[char] = new_node_idx
        # Move to the child node
        node_idx = trie_nodes[node_idx].children[char]
    # Mark the end of the string at the final node
    trie_nodes[node_idx].end_of_string_count += 1

# Recursive function to calculate descendant_count for a subtree rooted at node_idx
# This performs a post-order traversal implicitly. It modifies the trie_nodes list.
def calculate_descendant_count_recursive(node_idx):
    node = trie_nodes[node_idx]

    # The count for the current node is its own end_of_string_count
    # plus the descendant counts of all its children.
    count = node.end_of_string_count
    for child_idx in node.children.values():
        # Recursively call for children and add their counts
        count += calculate_descendant_count_recursive(child_idx)

    # Store the calculated descendant count in the node
    node.descendant_count = count
    # Return the count (used by the parent call)
    return count

# Recursive function to calculate the total LCP sum
# This performs a traversal (like DFS) and sums up contributions based on descendant counts.
# It reads from the trie_nodes list (specifically descendant_count).
def calculate_total_lcp_sum_recursive(node_idx, depth):
    node = trie_nodes[node_idx]

    total_sum = 0

    # According to the derived formula, the total sum is sum(C(k, 2))
    # over all nodes `u` in the trie with depth(u) >= 1, where k = descendant_count(u).
    # This node `u` is `trie_nodes[node_idx]`, its depth is `depth`.
    # The root node has depth 0 and represents the empty prefix, which does not contribute
    # to the LCP length calculation as per sample outputs (f("ab", "arc")=1, not including empty string).
    if depth >= 1:
        k = node.descendant_count
        # We only add contributions for nodes that represent a common prefix
        # for at least two strings (k >= 2). C(k, 2) = k * (k - 1) / 2
        if k >= 2:
            total_sum += k * (k - 1) // 2

    # Recurse on children and add their contributions to the sum
    for child_idx in node.children.values():
        total_sum += calculate_total_lcp_sum_recursive(child_idx, depth + 1)

    return total_sum

# Main function to solve the problem
def solve():
    # Read the number of strings
    N = int(sys.stdin.readline())

    # Initialize the trie with the root node
    # The root node is at index 0 in the trie_nodes list
    trie_nodes.append(TrieNode())

    # Read each string and insert into the trie
    # This builds the trie structure by adding nodes to trie_nodes list
    for _ in range(N):
        s = sys.stdin.readline().strip()
        insert(s)

    # Increase the recursion depth limit.
    # The maximum depth of recursive calls (for calculate_descendant_count and calculate_total_lcp_sum)
    # is the height of the trie. The height is at most the maximum string length.
    # The maximum total length of all strings is 3 * 10^5.
    # The number of nodes in the trie is at most 1 + sum(|S_i|), which is at most 300001.
    # Setting the limit based on the number of nodes (which is len(trie_nodes) after building)
    # plus a small buffer is a safe approach.
    # Ensure the limit is set before the recursive calls.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(trie_nodes) + 100))

    # Calculate the descendant count for all nodes in a post-order traversal.
    # This updates the 'descendant_count' attribute for each TrieNode in trie_nodes.
    # Start the recursive calculation from the root node (index 0).
    # The return value of the root call is N, but we don't explicitly need it here.
    calculate_descendant_count_recursive(0)

    # Calculate the total sum of LCPs.
    # This performs a traversal (like DFS) and sums up contributions from nodes with depth >= 1.
    # Start the recursive calculation from the root node (index 0) at depth 0.
    total_lcp_sum = calculate_total_lcp_sum_recursive(0, 0)

    # Print the final result
    print(total_lcp_sum)

# Ensure the solve function is called when the script is executed
if __name__ == "__main__":
    solve()