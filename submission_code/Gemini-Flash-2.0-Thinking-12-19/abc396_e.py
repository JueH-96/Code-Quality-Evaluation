import sys

# Increasing recursion depth for potentially deep DSU structure before path compression
# N <= 2e5, set limit slightly above N
sys.setrecursionlimit(200005)

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        # diff[i] stores the XOR difference between the value of node i and its parent
        # Specifically, a_i ^ a_parent[i] = diff[i]
        self.diff = [0] * n

    def find(self, i):
        """
        Finds the root of node i and the XOR difference between a_i and a_root.
        Returns (root, a_i ^ a_root).
        """
        if self.parent[i] == i:
            return (i, 0)
        
        # Path compression with difference update
        root, d_parent = self.find(self.parent[i])
        # We have:
        # 1. a_i ^ a_parent[i] = self.diff[i] (current stored difference)
        # 2. a_parent[i] ^ a_root = d_parent (difference from recursive call)
        # We want a_i ^ a_root. XORing (1) and (2):
        # (a_i ^ a_parent[i]) ^ (a_parent[i] ^ a_root) = self.diff[i] ^ d_parent
        # a_i ^ a_root = self.diff[i] ^ d_parent
        self.diff[i] ^= d_parent
        self.parent[i] = root
        return (root, self.diff[i])

    def union(self, u, v, weight):
        """
        Processes constraint a_u ^ a_v = weight.
        Merges the sets containing u and v.
        Returns True if consistent, False if inconsistent.
        """
        root_u, d_u = self.find(u)
        root_v, d_v = self.find(v)

        if root_u != root_v:
            # Merge root_v into root_u's component
            self.parent[root_v] = root_u
            
            # We need to set self.diff[root_v] such that a_root_v ^ a_root_u = self.diff[root_v].
            # We know:
            # a_u ^ a_v = weight (constraint)
            # a_u ^ a_root_u = d_u => a_u = a_root_u ^ d_u
            # a_v ^ a_root_v = d_v => a_v = a_root_v ^ d_v
            # Substitute into constraint: (a_root_u ^ d_u) ^ (a_root_v ^ d_v) = weight
            # a_root_u ^ a_root_v = weight ^ d_u ^ d_v
            # So, a_root_v ^ a_root_u = weight ^ d_u ^ d_v.
            self.diff[root_v] = d_u ^ d_v ^ weight
            return True
        else:
            # u and v are already in the same component. Check if constraint is consistent.
            # The current difference between u and v is a_u ^ a_v.
            # From find: a_u ^ a_root_u = d_u and a_v ^ a_root_v = d_v.
            # Since root_u == root_v, we have a_v ^ a_root_u = d_v.
            # So, a_u ^ a_v = (a_u ^ a_root_u) ^ (a_v ^ a_root_u) = d_u ^ d_v.
            # Consistency requires a_u ^ a_v = weight.
            return (d_u ^ d_v) == weight

def solve():
    # Read input using sys.stdin.readline for speed
    input = sys.stdin.readline
    N, M = map(int, input().split())
    constraints = []
    for _ in range(M):
        u, v, z = map(int, input().split())
        # Adjust to 0-based indexing
        constraints.append((u - 1, v - 1, z))

    # A will store the final sequence of non-negative integers
    A = [0] * N

    # Maximum Z_i is 10^9. $2^{29} < 10^9 < 2^{30}$.
    # We need bits up to index 29. Iterate through bit positions k from 0 to 29.
    max_bits = 30 # Process bits 0 through 29

    for k in range(max_bits):
        dsu = DSU(N)
        is_possible_bit_k = True

        # Process constraints for the current bit k
        for u, v, z in constraints:
            z_k = (z >> k) & 1 # k-th bit of Z_i
            if not dsu.union(u, v, z_k):
                # Inconsistent constraint found for this bit. No solution exists.
                is_possible_bit_k = False
                break # No need to process further constraints for this bit

        if not is_possible_bit_k:
            print(-1)
            return # Exit the program as no good sequence exists

        # If bit k is possible, determine the optimal bit values to minimize sum.
        # We need to decide the value (0 or 1) for the root of each component
        # to minimize the sum of bits in that component for this bit position k.

        # Calculate counts of nodes with difference 0 and 1 relative to their root
        # for each component.
        root_counts = {} # Map root_index -> [count_diff_0, count_diff_1]

        # Iterate through all nodes to find their roots and calculate difference counts.
        # Calling find(j) ensures path compression is applied and diff[j] is
        # correctly updated to be a_{j,k} ^ a_{root_j, k}.
        for j in range(N):
             root_j, d_j = dsu.find(j) # d_j is a_{j,k} ^ a_{root_j, k}
             if root_j not in root_counts:
                 root_counts[root_j] = [0, 0]
             # d_j is 0 if a_{j,k} == a_{root_j, k}
             # d_j is 1 if a_{j,k} != a_{root_j, k}
             # Store count based on d_j value
             root_counts[root_j][d_j] += 1

        # Determine the optimal bit value for each root
        root_bit_values = {} # Map root_index -> optimal_bit_value (0 or 1)
        for root in root_counts:
            c0, c1 = root_counts[root]
            # If we set a_{root, k} = 0:
            #   Nodes j in this component with a_{j,k} ^ a_{root,k} = 0 (c0 nodes) will have a_{j,k} = 0.
            #   Nodes j in this component with a_{j,k} ^ a_{root,k} = 1 (c1 nodes) will have a_{j,k} = 1.
            #   Sum of bits in this component = 0 * c0 + 1 * c1 = c1.
            # If we set a_{root, k} = 1:
            #   Nodes j in this component with a_{j,k} ^ a_{root,k} = 0 (c0 nodes) will have a_{j,k} = 1.
            #   Nodes j in this component with a_{j,k} ^ a_{root,k} = 1 (c1 nodes) will have a_{j,k} = 0.
            #   Sum of bits in this component = 1 * c0 + 0 * c1 = c0.
            # Choose the root bit value that gives the minimum sum of bits for this component.
            if c1 <= c0:
                root_bit_values[root] = 0
            else:
                root_bit_values[root] = 1

        # Determine the k-th bit value for each node j (a_{j,k}) and update A[j]
        for j in range(N):
            # After the find calls in the root_counts loop, parent[j] is the root,
            # and diff[j] is the difference a_{j,k} ^ a_{root_j, k}.
            root_j = dsu.parent[j]
            d_j = dsu.diff[j] # d_j = a_{j,k} ^ a_{root_j, k}

            # We know a_{j,k} ^ a_{root_j, k} = d_j.
            # We chose a_{root_j, k} = root_bit_values[root_j].
            # So, a_{j,k} = a_{root_j, k} ^ d_j = root_bit_values[root_j] ^ d_j.
            final_a_j_k = root_bit_values[root_j] ^ d_j

            # If the k-th bit of A[j] should be 1, set the k-th bit in A[j].
            if final_a_j_k == 1:
                A[j] |= (1 << k) # Equivalent to A[j] = A[j] | (1 << k)

    # All bits processed. A contains the resulting sequence that minimizes the sum.
    # Print the sequence elements separated by spaces.
    # Use sys.stdout.write for potentially faster output, joining elements with spaces.
    sys.stdout.write(" ".join(map(str, A)) + "
")

solve()