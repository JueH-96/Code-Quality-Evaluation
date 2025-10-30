import sys

# It's good practice in competitive programming to use sys.stdin.readline for faster input
# when N or M are large, and sys.stdout.write for faster output.
input = sys.stdin.readline

class SegmentTree:
    """
    A Segment Tree designed to find the minimum value in a range and, more specifically,
    to find the index of the first element (from left to right) in a range
    that satisfies a given condition (value <= target_B).
    """
    def __init__(self, arr):
        """
        Initializes the SegmentTree with the given array.
        arr: The input array (A_i gourmet levels). Assumed to be 0-indexed.
        """
        self.n = len(arr)
        # The tree array requires about 4 * N space for safety in 1-indexed tree representation.
        # Initialized with float('inf') for minimum queries.
        self.tree = [float('inf')] * (4 * self.n)
        self.arr = arr  # Store the original array A for building the tree.

        # Build the segment tree starting from node 1, covering the full array range [0, N-1].
        self._build(1, 0, self.n - 1)

    def _build(self, node_idx, L, R):
        """
        Recursively builds the segment tree. Each node stores the minimum value
        in the range it covers.
        
        node_idx: The current node's index in the self.tree array.
        L, R: The start and end indices (inclusive) of the segment covered by this node
              in the original array (self.arr).
        """
        if L == R:
            # If it's a leaf node, store the actual array value.
            self.tree[node_idx] = self.arr[L]
        else:
            # For non-leaf nodes, recursively build children and then store
            # the minimum of the children's values.
            mid = (L + R) // 2
            self._build(2 * node_idx, L, mid)          # Build left child (covers [L, mid])
            self._build(2 * node_idx + 1, mid + 1, R)  # Build right child (covers [mid+1, R])
            self.tree[node_idx] = min(self.tree[2 * node_idx], self.tree[2 * node_idx + 1])

    def find_first_eater(self, node_idx, current_L, current_R, target_B):
        """
        Finds the 1-indexed person ID of the first person (smallest original index)
        in the segment tree's covered range [current_L, current_R] whose gourmet level
        (A_i) is less than or equal to target_B (sushi deliciousness).

        This function performs a specialized query that prioritizes finding the
        leftmost matching element.

        node_idx: The current node's index.
        current_L, current_R: The segment range covered by the current node.
        target_B: The deliciousness of the sushi we are trying to find an eater for.
        
        Returns:
            The 1-indexed person ID if an eater is found, otherwise -1.
        """
        # Base Case 1: If the minimum gourmet level in this segment is greater than target_B,
        # then no one in this range can eat the sushi. Prune this branch.
        if self.tree[node_idx] > target_B:
            return -1

        # Base Case 2: If this is a leaf node, it means we have found the leftmost
        # person who can eat the sushi (due to the recursive search order).
        # current_L is the 0-indexed person index.
        if current_L == current_R:
            return current_L + 1  # Convert to 1-indexed person ID

        # Recursive Step: This is a non-leaf node.
        # We need to check the left child first to find the "first" person.
        mid = (current_L + current_R) // 2
        
        # Check if the left child's range potentially contains a person who can eat.
        # If the minimum in the left child's segment is <= target_B, then an eater
        # *might* be in the left half. Recursively search there.
        if self.tree[2 * node_idx] <= target_B:
            result = self.find_first_eater(2 * node_idx, current_L, mid, target_B)
            # If an eater was found in the left subtree, return it immediately.
            if result != -1:
                return result
        
        # If no eater was found in the left subtree (either because min was too high,
        # or recursive call returned -1), then we must check the right subtree.
        # It's guaranteed that if an eater exists in the current node's total range,
        # and not in the left child's range, it must be in the right child's range.
        return self.find_first_eater(2 * node_idx + 1, mid + 1, current_R, target_B)


def solve():
    """
    Main function to read input, perform the sushi distribution simulation,
    and print the results.
    """
    N, M = map(int, input().split())
    A = list(map(int, input().split()))  # Gourmet levels of N people (0-indexed internally)
    B = list(map(int, input().split()))  # Deliciousness of M sushi pieces

    # Create the Segment Tree based on the gourmet levels.
    # The segment tree will allow efficient queries for the first person who can eat a sushi.
    seg_tree = SegmentTree(A)

    # Store results for each sushi piece.
    results = []
    for sushi_deliciousness in B:
        # Query the segment tree to find the 1-indexed ID of the first person
        # who can eat the current sushi.
        # The query starts from the root (node 1) and covers the entire person range [0, N-1].
        eater_id = seg_tree.find_first_eater(1, 0, N - 1, sushi_deliciousness)
        results.append(eater_id)

    # Print all the results, each on a new line.
    for res in results:
        sys.stdout.write(str(res) + '
')

# Call the solve function to run the program.
solve()