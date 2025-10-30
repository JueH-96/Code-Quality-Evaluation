import sys

# Define Node class for Segment Tree
class Node:
    def __init__(self):
        self._min = float('inf')  # minimum value in this range
        self._sum = 0             # sum of values in this range (not strictly needed but can be helpful for debugging)
        self._num_zero = 0        # count of zeros in this range
        self._lazy_sub = 0        # value to subtract from all elements in this range (lazy tag)

class SegmentTree:
    def __init__(self, N, A):
        self.N = N
        # The tree list will hold Node objects. 4*N is a common safe size for a segment tree.
        self.tree = [Node() for _ in range(4 * N)] 
        self._build(1, 0, N - 1, A)

    def _build(self, node_idx, L, R, A):
        """
        Recursively builds the segment tree.
        node_idx: current node's index in self.tree
        L, R: range covered by current node
        A: initial array of stone counts
        """
        if L == R: # Leaf node
            self.tree[node_idx]._min = A[L]
            self.tree[node_idx]._sum = A[L]
            self.tree[node_idx]._num_zero = 1 if A[L] == 0 else 0
            return

        mid = (L + R) // 2
        self._build(2 * node_idx, L, mid, A)         # Build left child
        self._build(2 * node_idx + 1, mid + 1, R, A) # Build right child
        self._recalculate(node_idx) # Update parent node based on children

    def _recalculate(self, node_idx):
        """
        Updates the values (_min, _sum, _num_zero) of a parent node
        based on its children's values. Assumes children are up-to-date.
        """
        left_child = self.tree[2 * node_idx]
        right_child = self.tree[2 * node_idx + 1]
        
        self.tree[node_idx]._min = min(left_child._min, right_child._min)
        self.tree[node_idx]._sum = left_child._sum + right_child._sum
        self.tree[node_idx]._num_zero = left_child._num_zero + right_child._num_zero

    def _apply_lazy(self, node_idx, val, count_elements):
        """
        Applies a lazy subtraction to a node's values.
        val: amount to subtract
        count_elements: number of elements in the node's range
        """
        self.tree[node_idx]._sum -= val * count_elements
        self.tree[node_idx]._min -= val
        self.tree[node_idx]._lazy_sub += val

    def _push_down(self, node_idx, L, R):
        """
        Pushes down lazy tags from current node to its children.
        """
        if self.tree[node_idx]._lazy_sub > 0 and L != R: # Only push down if not a leaf and has lazy tag
            mid = (L + R) // 2
            
            # Apply lazy tag to left child
            self._apply_lazy(2 * node_idx, self.tree[node_idx]._lazy_sub, mid - L + 1)
            # Apply lazy tag to right child
            self._apply_lazy(2 * node_idx + 1, self.tree[node_idx]._lazy_sub, R - (mid + 1) + 1)
            
            self.tree[node_idx]._lazy_sub = 0 # Reset lazy tag for current node

    def range_decrement(self, query_L, query_R, delta=1):
        """
        Decrements values in the range [query_L, query_R] by delta,
        clamping values at 0 (i.e., val = max(0, val - delta)).
        """
        if query_L > query_R: # Invalid or empty range
            return
        self._range_decrement_recursive(1, 0, self.N - 1, query_L, query_R, delta)

    def _range_decrement_recursive(self, node_idx, L, R, query_L, query_R, delta):
        # Current node range [L, R], query range [query_L, query_R]
        
        # Case 1: Current node range is completely outside query range
        if query_L > R or query_R < L:
            return
        
        # Push down lazy tag before processing current node's own state or recursing
        self._push_down(node_idx, L, R) 
        
        # Case 2: Current node is a leaf node
        if L == R:
            # Only decrement if the value is positive and the query range includes this leaf
            if self.tree[node_idx]._min > 0 and query_L <= L and R <= query_R: 
                self.tree[node_idx]._min = max(0, self.tree[node_idx]._min - delta)
                self.tree[node_idx]._sum = self.tree[node_idx]._min
                self.tree[node_idx]._num_zero = 1 if self.tree[node_idx]._min == 0 else 0
            return
        
        # Case 3: All values in current node's range are greater than delta. Apply lazy tag.
        # This implies no value will become 0 from this operation within this range.
        if self.tree[node_idx]._min > delta:
            self._apply_lazy(node_idx, delta, R - L + 1)
            return

        # Case 4: Current node range partially overlaps query range or min_val <= delta (meaning some values might become 0). Recurse.
        mid = (L + R) // 2
        self._range_decrement_recursive(2 * node_idx, L, mid, query_L, query_R, delta)
        self._range_decrement_recursive(2 * node_idx + 1, mid + 1, R, query_L, query_R, delta)
        
        self._recalculate(node_idx) # Update parent node after children are processed

    def query_num_positive(self, query_L, query_R):
        """
        Queries the number of positive values in the range [query_L, query_R].
        """
        if query_L > query_R: # Invalid or empty range
            return 0
        return self._query_num_positive_recursive(1, 0, self.N - 1, query_L, query_R)

    def _query_num_positive_recursive(self, node_idx, L, R, query_L, query_R):
        self._push_down(node_idx, L, R) # Push down lazy tag first
        
        # Case 1: Current node range is completely outside query range
        if query_L > R or query_R < L:
            return 0
        
        # Case 2: Current node range is completely inside query range
        if query_L <= L and R <= query_R:
            return (R - L + 1) - self.tree[node_idx]._num_zero

        # Case 3: Current node range partially overlaps query range. Recurse.
        mid = (L + R) // 2
        left_res = self._query_num_positive_recursive(2 * node_idx, L, mid, query_L, query_R)
        right_res = self._query_num_positive_recursive(2 * node_idx + 1, mid + 1, R, query_L, query_R)
        return left_res + right_res

    def point_update(self, target_idx, new_val):
        """
        Updates the value at target_idx to new_val.
        """
        self._point_update_recursive(1, 0, self.N - 1, target_idx, new_val)

    def _point_update_recursive(self, node_idx, L, R, target_idx, new_val):
        self._push_down(node_idx, L, R) # Push down lazy tag first
        
        if L == R: # Leaf node
            self.tree[node_idx]._min = new_val
            self.tree[node_idx]._sum = new_val
            self.tree[node_idx]._num_zero = 1 if new_val == 0 else 0
            return

        mid = (L + R) // 2
        if target_idx <= mid:
            self._point_update_recursive(2 * node_idx, L, mid, target_idx, new_val)
        else:
            self._point_update_recursive(2 * node_idx + 1, mid + 1, R, target_idx, new_val)
        
        self._recalculate(node_idx) # Update parent node after children are processed

    def _get_leaf_value(self, node_idx, L, R, target_idx):
        """
        Helper to get the final value of a specific leaf node.
        """
        self._push_down(node_idx, L, R) # Ensure lazy tags are applied down to this path
        if L == R:
            return self.tree[node_idx]._sum # _sum and _min are identical for leaf nodes
        
        mid = (L + R) // 2
        if target_idx <= mid:
            return self._get_leaf_value(2 * node_idx, L, mid, target_idx)
        else:
            return self._get_leaf_value(2 * node_idx + 1, mid + 1, R, target_idx)

# Main part of the program
def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    seg_tree = SegmentTree(N, A)

    # Simulate the process year by year
    for k in range(N): # Alien k (0-indexed) becomes adult
        # 1. Count givers among aliens 0 to k-1
        # The range for givers is [0, k-1]. If k=0, this range is empty, query_num_positive handles this.
        num_givers = seg_tree.query_num_positive(0, k - 1)
        
        # 2. Alien k receives stones
        # Get current stones of alien k before adding gifts.
        current_stones_k = seg_tree._get_leaf_value(1, 0, N - 1, k) 
        # Update alien k's stone count.
        seg_tree.point_update(k, current_stones_k + num_givers)
        
        # 3. Givers (aliens 0 to k-1) lose one stone each (if positive)
        seg_tree.range_decrement(0, k - 1, 1)

    # Collect final stone counts for all aliens
    final_stones = [0] * N
    for i in range(N):
        final_stones[i] = seg_tree._get_leaf_value(1, 0, N - 1, i)

    # Print the result separated by spaces
    print(*(final_stones))

# Call the solve function
solve()