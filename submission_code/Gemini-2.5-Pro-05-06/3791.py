from typing import List

class Solution:
    # self.tree: List[int] will store the segment tree.
    # self.n_seg: int will store the number of elements (baskets),
    #              and serve as a "not found" sentinel for _query.

    def _build(self, node_idx: int, L: int, R: int, baskets_arr: List[int]):
        """
        Builds the segment tree.
        node_idx: current node's index in self.tree (1-indexed).
        L, R: range [L, R] (0-indexed) covered by this node.
        baskets_arr: the initial array of basket capacities.
        """
        if L == R:
            # Leaf node: store the capacity of the basket at index L
            self.tree[node_idx] = baskets_arr[L]
        else:
            M = (L + R) // 2
            left_child_idx = 2 * node_idx
            right_child_idx = 2 * node_idx + 1
            
            self._build(left_child_idx, L, M, baskets_arr)
            self._build(right_child_idx, M + 1, R, baskets_arr)
            
            # Internal node: store the maximum capacity in its range
            self.tree[node_idx] = max(self.tree[left_child_idx], self.tree[right_child_idx])

    def _update(self, node_idx: int, L: int, R: int, target_idx: int, new_val: int):
        """
        Updates the capacity of a basket at target_idx and propagates changes up the tree.
        target_idx: 0-indexed original index of the basket to update.
        new_val: new capacity value (e.g., -1 to mark as used).
        """
        if L == R:
            # Leaf node corresponding to target_idx
            self.tree[node_idx] = new_val
        else:
            M = (L + R) // 2
            left_child_idx = 2 * node_idx
            right_child_idx = 2 * node_idx + 1
            
            if target_idx <= M:
                # target_idx is in the left child's range
                self._update(left_child_idx, L, M, target_idx, new_val)
            else:
                # target_idx is in the right child's range
                self._update(right_child_idx, M + 1, R, target_idx, new_val)
            
            # Update this node's max capacity based on its children
            self.tree[node_idx] = max(self.tree[left_child_idx], self.tree[right_child_idx])

    def _query(self, node_idx: int, L: int, R: int, K_fruit_qty: int) -> int:
        """
        Finds the leftmost basket index j in range [L,R] such that its current
        capacity (stored in the tree) is >= K_fruit_qty.
        K_fruit_qty: the required minimum capacity.
        Returns: 0-indexed basket index j, or self.n_seg if no such basket is found.
        """
        # If max capacity in this node's range is less than required, no solution here
        if self.tree[node_idx] < K_fruit_qty:
            return self.n_seg  # "not found" sentinel
        
        # If this is a leaf node
        if L == R:
            # And self.tree[node_idx] >= K_fruit_qty (implicit from the first check),
            # this basket L is suitable.
            return L 
            
        M = (L + R) // 2
        left_child_idx = 2 * node_idx
        right_child_idx = 2 * node_idx + 1
        
        # Try left child first:
        # If the maximum capacity in the left child's range is sufficient...
        if self.tree[left_child_idx] >= K_fruit_qty:
            res_left = self._query(left_child_idx, L, M, K_fruit_qty)
            # If a suitable basket is found in the left subtree, it's the leftmost.
            if res_left != self.n_seg:
                return res_left
        
        # If not found in left child's range (either its max was too small, or recursive call returned not_found),
        # then try right child's range.
        # The check self.tree[right_child_idx] >= K_fruit_qty for the right child
        # will be handled by its own recursive call's initial check.
        res_right = self._query(right_child_idx, M + 1, R, K_fruit_qty)
        return res_right

    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        
        # Constraints: n == fruits.length == baskets.length, 1 <= n <= 10^5.
        # So, n=0 case is not expected. If it were, `if n == 0: return 0` would be needed.
        
        self.n_seg = n  # Store n for use in helper methods, also as "not found" sentinel.
        
        # Initialize segment tree array.
        # Capacities are >= 1. We use -1 to mark used baskets.
        # So, -1 is a good default for tree elements (represents no available positive capacity).
        self.tree = [-1] * (4 * n) 
        
        # Build segment tree from initial basket capacities
        # Root is at index 1, covering original indices [0, n-1].
        self._build(1, 0, n - 1, baskets)

        unplaced_count = 0
        for fruit_qty in fruits:
            # Find the leftmost available basket with capacity >= fruit_qty
            basket_idx = self._query(1, 0, n - 1, fruit_qty)
            
            if basket_idx == self.n_seg:  # self.n_seg indicates "not found"
                unplaced_count += 1
            else:
                # Mark the chosen basket as used by setting its capacity to -1
                # This value is less than any possible fruit_qty (>=1).
                self._update(1, 0, n - 1, basket_idx, -1)
        
        return unplaced_count