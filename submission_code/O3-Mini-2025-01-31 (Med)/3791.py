from typing import List

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(baskets)
        # Build a segment tree for the baskets capacities.
        # The tree will allow us to quickly find the leftmost index with capacity >= x.
        size = 1
        while size < n:
            size *= 2
        # Initialize tree with 0; tree indices [size, size+n) hold baskets values.
        tree = [0] * (2 * size)
        
        # Build the leaves
        for i in range(n):
            tree[size + i] = baskets[i]
        # For leaves that don't correspond to baskets, they remain 0.
        # Build the segment tree: parent's value = max(child_1, child_2)
        for i in range(size - 1, 0, -1):
            tree[i] = max(tree[2*i], tree[2*i + 1])
        
        # Function to update basket at position pos to new_value.
        def update(pos: int, new_value: int):
            pos += size
            tree[pos] = new_value
            pos //= 2
            while pos:
                tree[pos] = max(tree[2*pos], tree[2*pos+1])
                pos //= 2
        
        # Function to query for the leftmost index with value >= x in the segment tree.
        def query(x: int) -> int:
            # If the maximum in whole tree is less than x, then no basket qualifies.
            if tree[1] < x:
                return -1
            idx = 1
            # Descend the tree to find the leftmost index
            while idx < size:  # while not a leaf
                if tree[2*idx] >= x:
                    idx = 2*idx
                else:
                    idx = 2*idx+1
            return idx - size
        
        unplaced = 0
        
        for req in fruits:
            basket_index = query(req)
            if basket_index == -1:
                unplaced += 1
            else:
                # Mark the basket as used by setting it to 0.
                update(basket_index, 0)
        
        return unplaced