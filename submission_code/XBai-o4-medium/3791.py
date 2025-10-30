import bisect
from typing import List

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.min_tree = [float('inf')] * (2 * self.size)
        # Fill the leaves
        for i in range(self.n):
            self.min_tree[self.size + i] = data[i]
        # Build the tree
        for i in range(self.size - 1, 0, -1):
            self.min_tree[i] = min(self.min_tree[2*i], self.min_tree[2*i+1])
    
    def update(self, pos, value):
        # pos is 0-based in the original data
        pos += self.size
        self.min_tree[pos] = value
        pos >>= 1
        while pos >= 1:
            new_val = min(self.min_tree[2*pos], self.min_tree[2*pos+1])
            if self.min_tree[pos] == new_val:
                break
            self.min_tree[pos] = new_val
            pos >>= 1
    
    def query_min(self, l, r):
        # l and r are 0-based in the original data
        res = float('inf')
        l += self.size
        r += self.size
        while l <= r:
            if l % 2 == 1:
                res = min(res, self.min_tree[l])
                l += 1
            if r % 2 == 0:
                res = min(res, self.min_tree[r])
                r -= 1
            l >>= 1
            r >>= 1
        return res

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        # Prepare sorted_baskets
        sorted_baskets = sorted( [ (baskets[i], i) for i in range(n) ], key=lambda x: (x[0], x[1]) )
        sorted_capacities = [x[0] for x in sorted_baskets]
        original_to_pos = {i: pos for pos, (cap, i) in enumerate(sorted_baskets)}
        # Prepare data for segment tree: original indexes
        data = [i for (cap, i) in sorted_baskets]
        st = SegmentTree(data)
        unplaced = 0
        for q in fruits:
            # Find start index
            start = bisect.bisect_left(sorted_capacities, q)
            if start >= n:
                unplaced += 1
                continue
            # Query the segment tree from start to n-1 (original data 0-based)
            min_orig = st.query_min(start, n-1)
            if min_orig == float('inf'):
                unplaced += 1
            else:
                # Find the position in the sorted_baskets list
                pos_in_sorted = original_to_pos[min_orig]
                # Update the segment tree to mark this position as used
                st.update(pos_in_sorted, float('inf'))
        return unplaced