from typing import List

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(baskets)
        # We'll build an iterative segment tree to support:
        # 1. Query: find the leftmost basket (by index) with capacity >= target.
        # 2. Update: mark a basket as used by setting its capacity to 0.
        # Since all basket capacities are at least 1, using 0 as "used" is safe.
        
        # Determine size for the complete binary tree (power of 2)
        size = 1
        while size < n:
            size *= 2
        
        # Create the tree array (2 * size) with initial values.
        tree = [0] * (2 * size)
        # Set the leaves to the baskets' capacities.
        for i in range(n):
            tree[size + i] = baskets[i]
        # Build the tree: each parent stores the maximum capacity available in that segment.
        for i in range(size - 1, 0, -1):
            tree[i] = max(tree[2 * i], tree[2 * i + 1])
        
        # Update function to mark a basket as used (set its capacity to 0).
        def update(idx: int, value: int):
            pos = size + idx
            tree[pos] = value
            pos //= 2
            while pos:
                tree[pos] = max(tree[2 * pos], tree[2 * pos + 1])
                pos //= 2
        
        # Query function: Find the leftmost basket (i.e. smallest index) that has capacity >= target.
        # If no such basket exists, return -1.
        def query(target: int) -> int:
            # If the root's maximum capacity is less than target, no basket can fit the fruit.
            if tree[1] < target:
                return -1
            pos = 1
            # Traverse down the tree to find the leftmost suitable basket.
            while pos < size:  # until a leaf node is reached
                if tree[2 * pos] >= target:
                    pos = 2 * pos
                else:
                    pos = 2 * pos + 1
            return pos - size  # convert tree index back to basket index
        
        unplaced = 0
        # Process each fruit type in order.
        for fruit in fruits:
            basket_idx = query(fruit)
            if basket_idx == -1:
                # If no basket can hold this fruit type, count it as unplaced.
                unplaced += 1
            else:
                # Mark the used basket as unavailable.
                update(basket_idx, 0)
        
        return unplaced