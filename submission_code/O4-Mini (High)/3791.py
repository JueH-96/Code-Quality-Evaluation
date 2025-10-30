from typing import List

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        # Build a segment tree over basket capacities to support:
        # 1) finding the leftmost index with capacity >= f
        # 2) updating that capacity to 0 once used
        size = 1
        while size < n:
            size <<= 1
        # tree[i] will hold the maximum capacity in that segment
        tree = [0] * (2 * size)
        # Initialize leaves
        for i in range(n):
            tree[size + i] = baskets[i]
        # Build internal nodes
        for i in range(size - 1, 0, -1):
            left = tree[2 * i]
            right = tree[2 * i + 1]
            tree[i] = left if left > right else right

        unplaced = 0
        # Process each fruit
        for f in fruits:
            # If the overall max capacity is less than f, can't place this fruit
            if tree[1] < f:
                unplaced += 1
                continue
            # Otherwise, descend the tree to find the leftmost index with cap >= f
            idx = 1
            while idx < size:
                if tree[2 * idx] >= f:
                    idx = 2 * idx
                else:
                    idx = 2 * idx + 1
            # idx now points to the leaf; compute the basket index
            basket_index = idx - size
            # Mark this basket as used by setting capacity to 0
            tree[idx] = 0
            # Update ancestors
            parent = idx >> 1
            while parent:
                left = tree[2 * parent]
                right = tree[2 * parent + 1]
                tree[parent] = left if left > right else right
                parent >>= 1

        return unplaced