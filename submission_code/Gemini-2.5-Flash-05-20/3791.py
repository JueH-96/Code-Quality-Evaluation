import math
from typing import List

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        
        # Segment tree implementation
        # self.tree stores the maximum capacity in its range for each node.
        # A size of 4*n is generally sufficient for a segment tree to avoid index out of bounds.
        self.tree = [0] * (4 * n)
        # Store original baskets capacities. The segment tree will manage the 'used' state.
        self.baskets_original = baskets 
        
        # Helper function to build the segment tree
        # node: current node index in the self.tree array
        # start, end: the range [start, end] of original baskets indices this node covers
        def build(node: int, start: int, end: int):
            if start == end:
                # Leaf node: store the capacity of the basket at this index
                self.tree[node] = self.baskets_original[start]
            else:
                mid = (start + end) // 2
                # Recursively build left and right children
                build(2 * node, start, mid)
                build(2 * node + 1, mid + 1, end)
                # Internal node: store the maximum capacity of its children's ranges
                self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])

        # Helper function to update a basket's capacity in the segment tree
        # This is used to mark a basket as 'used' by setting its capacity to 0.
        # node: current node index
        # start, end: current range [start, end]
        # idx: the specific index of the basket to update
        # val: the new value for the basket capacity (0 to mark as used)
        def update(node: int, start: int, end: int, idx: int, val: int):
            if start == end:
                # Found the leaf node corresponding to 'idx', update its value
                self.tree[node] = val
            else:
                mid = (start + end) // 2
                if start <= idx <= mid:
                    # 'idx' is in the left child's range, recurse left
                    update(2 * node, start, mid, idx, val)
                else:
                    # 'idx' is in the right child's range, recurse right
                    update(2 * node + 1, mid + 1, end, idx, val)
                # After updating a child, update the current node's value
                # (it should always reflect the max of its children)
                self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])

        # Helper function to query the segment tree for the leftmost suitable basket
        # node: current node index
        # start, end: current range [start, end]
        # fruit_qty: the minimum capacity required for the basket
        # Returns the leftmost index of an available basket that can hold fruit_qty, or -1 if none found
        def query(node: int, start: int, end: int, fruit_qty: int) -> int:
            # If the maximum capacity in this segment (self.tree[node]) is less than fruit_qty,
            # it means no basket in this entire range can hold the fruit.
            if self.tree[node] < fruit_qty:
                return -1
            
            # If we are at a leaf node and its capacity is sufficient, this is a suitable basket.
            # Since we search left-to-right, this will be the leftmost suitable one in its original range.
            if start == end:
                return start
            
            mid = (start + end) // 2
            
            # First, try to find a suitable basket in the left child's range.
            # This prioritizes finding the leftmost basket.
            res = query(2 * node, start, mid, fruit_qty)
            
            if res != -1:
                # If a suitable basket was found in the left half, return its index.
                # This is guaranteed to be the leftmost available.
                return res
            else:
                # If no suitable basket was found in the left half, search in the right half.
                return query(2 * node + 1, mid + 1, end, fruit_qty)

        # Initialize the segment tree by building it from the root (node 1) covering the entire range [0, n-1]
        build(1, 0, n - 1)

        unplaced_count = 0
        # Process each fruit type from left to right
        for fruit_qty in fruits:
            # Find the leftmost available basket for the current fruit_qty
            found_basket_idx = query(1, 0, n - 1, fruit_qty)
            
            if found_basket_idx != -1:
                # If a suitable basket is found, mark it as used by updating its capacity to 0 in the tree.
                # Capacities are 1 or greater, so 0 effectively removes it from consideration.
                update(1, 0, n - 1, found_basket_idx, 0)
            else:
                # If no suitable basket is found, increment the count of unplaced fruits.
                unplaced_count += 1
                
        return unplaced_count