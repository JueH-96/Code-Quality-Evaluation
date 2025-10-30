from typing import List

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        # build a segment tree to maintain the max capacity among free baskets
        size = 1
        while size < n:
            size <<= 1
        # tree nodes are 1-indexed; leaves start at index `size`
        tree = [0] * (2 * size)
        
        # initialize leaves with basket capacities
        for i in range(n):
            tree[size + i] = baskets[i]
        # build internal nodes
        for i in range(size - 1, 0, -1):
            tree[i] = max(tree[2 * i], tree[2 * i + 1])
        
        # query for the leftmost basket with capacity >= f
        def query(f: int) -> int:
            if tree[1] < f:
                return -1
            idx = 1
            left, right = 0, size - 1
            # descend the tree
            while idx < size:
                mid = (left + right) // 2
                if tree[2 * idx] >= f:
                    # go left
                    idx = 2 * idx
                    right = mid
                else:
                    # go right
                    idx = 2 * idx + 1
                    left = mid + 1
            # `left` is the leaf index in [0, size-1]
            if left >= n:
                return -1
            return left
        
        # update basket at position `pos` to capacity 0 (mark used)
        def update(pos: int):
            p = size + pos
            tree[p] = 0
            p //= 2
            while p:
                tree[p] = max(tree[2 * p], tree[2 * p + 1])
                p //= 2
        
        unplaced = 0
        # process fruits in order
        for f in fruits:
            j = query(f)
            if j == -1:
                unplaced += 1
            else:
                update(j)
        
        return unplaced