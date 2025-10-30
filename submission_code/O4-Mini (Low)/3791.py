from typing import List

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(baskets)
        # Build a segment tree for range maximum queries over baskets' capacities
        size = 1
        while size < n:
            size <<= 1
        # Initialize tree with zeros (since capacities >= 1)
        tree = [0] * (2 * size)
        # Fill leaves
        for i in range(n):
            tree[size + i] = baskets[i]
        # Build internal nodes
        for i in range(size - 1, 0, -1):
            tree[i] = max(tree[2*i], tree[2*i + 1])

        # Find the leftmost index in [l, r] whose capacity >= x
        def find_first(idx: int, l: int, r: int, x: int) -> int:
            # If this segment cannot satisfy x, bail out
            if tree[idx] < x:
                return -1
            if l == r:
                # Leaf node
                return l
            mid = (l + r) // 2
            # Try left child first
            left_res = find_first(2*idx, l, mid, x)
            if left_res != -1:
                return left_res
            # Otherwise try right child
            return find_first(2*idx + 1, mid + 1, r, x)

        # Update position p to value v in the segment tree
        def update(p: int, v: int):
            pos = size + p
            tree[pos] = v
            pos //= 2
            while pos >= 1:
                tree[pos] = max(tree[2*pos], tree[2*pos + 1])
                pos //= 2

        unplaced = 0
        for f in fruits:
            # Query the tree for the leftmost basket with capacity >= f
            idx = find_first(1, 0, size - 1, f)
            if idx == -1 or idx >= n:
                # No valid basket
                unplaced += 1
            else:
                # Place fruit and mark basket as used (capacity = 0)
                update(idx, 0)

        return unplaced