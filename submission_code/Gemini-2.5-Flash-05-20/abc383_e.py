# YOUR CODE HERE
import sys

# DSU class with counts for A and B nodes
class DSU:
    def __init__(self, n):
        # parent[i] stores the parent of element i. If parent[i] == i, i is the root of its set.
        self.parent = list(range(n + 1))
        # unmatched_A_count[i] stores the current count of A-nodes within the component
        # rooted at 'i' that have not yet been matched with a B-node.
        self.unmatched_A_count = [0] * (n + 1)
        # unmatched_B_count[i] stores the current count of B-nodes within the component
        # rooted at 'i' that have not yet been matched with an A-node.
        self.unmatched_B_count = [0] * (n + 1)

    def find(self, i):
        # Path compression: make all nodes on the path point directly to the root.
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            # Store the current counts of the component roots BEFORE the parent pointers are changed.
            # These counts represent the 'excess' A or B nodes in each component that are
            # available for matching with nodes from other components.
            A_count_i = self.unmatched_A_count[root_i]
            B_count_i = self.unmatched_B_count[root_i]
            A_count_j = self.unmatched_A_count[root_j]
            B_count_j = self.unmatched_B_count[root_j]

            # Calculate the number of new pairs that can be formed by connecting these two components.
            # We greedily match A-nodes from component_i with B-nodes from component_j,
            # and A-nodes from component_j with B-nodes from component_i.
            matched_pairs = min(A_count_i, B_count_j) + min(A_count_j, B_count_i)
            
            # Perform the union operation: make one root the parent of the other.
            # For simplicity, we choose root_i to be the new parent.
            # For optimal DSU performance, one would typically union by size/rank to keep trees flat.
            self.parent[root_j] = root_i

            # Update the counts for the new combined component (now rooted at root_i).
            # The total number of A-nodes in the new component is the sum of A-nodes from both old components.
            # The total number of B-nodes is also summed.
            # Then, subtract the newly formed `matched_pairs` from both counts,
            # as these nodes are now 'matched' and no longer available for new external matches
            # at a higher cost.
            self.unmatched_A_count[root_i] = (A_count_i + A_count_j) - matched_pairs
            self.unmatched_B_count[root_i] = (B_count_i + B_count_j) - matched_pairs
            
            return matched_pairs
        return 0 # Components were already connected, no new matches at this weight.

def solve():
    N, M, K = map(int, sys.stdin.readline().split())

    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        edges.append((w, u, v)) # Store as (weight, u, v) for sorting

    A_list = list(map(int, sys.stdin.readline().split()))
    B_list = list(map(int, sys.stdin.readline().split()))

    # Sort edges by weight in ascending order. This is crucial for the greedy approach.
    edges.sort()

    dsu = DSU(N)
    
    # Initialize the DSU with initial counts of A and B nodes at their respective vertices.
    # Each A_node and B_node initially represents an 'unmatched' entity in its own component.
    for a_node in A_list:
        dsu.unmatched_A_count[a_node] += 1
    for b_node in B_list:
        dsu.unmatched_B_count[b_node] += 1

    total_min_path_sum = 0

    # Iterate through sorted edges.
    # When an edge (u, v, w) is processed, it connects two previously separate components.
    # Any A-node in one component can now reach any B-node in the other component with
    # a path whose maximum edge weight is 'w'. This 'w' is the minimum such weight
    # for these newly connected pairs.
    for w, u, v in edges:
        # Perform the union operation. The DSU's union method calculates and returns
        # the number of pairs that were newly matched due to this connection.
        num_matched = dsu.union(u, v)
        # Add the cost of these new matches to the total sum.
        total_min_path_sum += num_matched * w
    
    # Print the final calculated minimum sum.
    sys.stdout.write(str(total_min_path_sum) + "
")

# Call the solve function to run the program.
solve()