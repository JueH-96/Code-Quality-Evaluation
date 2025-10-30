import sys

# Use __slots__ for memory optimization in TrieNode
class TrieNode:
    __slots__ = ['children', 'min_len']
    def __init__(self):
        self.children = [None] * 26
        # Store the minimum length of a full string S_i that passes through this node.
        # Initialize with infinity.
        self.min_len = float('inf')

class Trie:
    __slots__ = ['root']
    def __init__(self):
        self.root = TrieNode()

    def query_and_insert(self, s):
        curr = self.root
        s_len = len(s)

        # Minimum cost initialized to the cost of transforming to empty string
        min_cost = s_len

        # --- Query part ---
        # We traverse the trie with the current string 's'.
        # At each node `query_curr` at `query_depth`, if `query_curr.min_len` is not infinity,
        # it means there exists at least one previously inserted string `S_i`
        # whose path goes through `query_curr`. This implies S_k and S_i share a
        # common prefix of length `query_depth`. The value `query_curr.min_len`
        # is the minimum length of such an S_i.
        # The cost to transform S_k to such an S_i is |S_k| + |S_i| - 2 * |LCPrefix(S_k, S_i)|
        # which is s_len + query_curr.min_len - 2 * query_depth.
        # We want the minimum cost over all possible S_i (< k) reachable via common prefixes.
        # By iterating through the prefixes of S_k (represented by nodes in the trie traversal)
        # and checking the minimum string length stored at those nodes, we cover all relevant S_i.
        query_curr = self.root
        query_depth = 0
        for char in s:
            index = ord(char) - ord('a')
            if query_curr.children[index] is not None:
                query_curr = query_curr.children[index]
                query_depth += 1
                
                # If any previously inserted string S_i passes through this node...
                if query_curr.min_len != float('inf'):
                    # ...S_k and S_i share a common prefix of length `query_depth`.
                    # The minimum length among such S_i is stored in query_curr.min_len.
                    cost_match = s_len + query_curr.min_len - 2 * query_depth
                    min_cost = min(min_cost, cost_match)
            else:
                # No more common prefix with existing strings along this path
                break

        result_cost = min_cost # Store the calculated min_cost before insertion modifies the trie

        # --- Insert part ---
        # Insert the current string 's' into the trie and update min_len for all nodes on the path.
        curr = self.root
        # The root represents the empty prefix. Update its min_len.
        curr.min_len = min(curr.min_len, s_len)

        # Traverse again to create nodes if necessary and update min_len
        for char in s:
            index = ord(char) - ord('a')
            if curr.children[index] is None:
                curr.children[index] = TrieNode()
            curr = curr.children[index]
            # Update min_len for the current node representing the prefix s[:depth]
            # The depth is implicitly tracked by the loop iteration, but the node itself
            # doesn't need to store it. We only need the min_len at the node.
            curr.min_len = min(curr.min_len, s_len)

        return result_cost

# Main execution
if __name__ == "__main__":
    # Read input
    N = int(sys.stdin.readline())
    
    # Initialize Trie
    trie = Trie()
    
    # Store results
    results = []

    # Process each string
    for _ in range(N):
        S_k = sys.stdin.readline().strip()
        cost = trie.query_and_insert(S_k)
        results.append(str(cost))

    # Print results
    sys.stdout.write("
".join(results) + "
")