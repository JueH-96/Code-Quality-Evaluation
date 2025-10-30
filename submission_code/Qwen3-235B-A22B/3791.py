from typing import List

class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        if n == 0:
            return 0
        
        class SegmentTree:
            def __init__(self, data):
                self.n = len(data)
                self.size = 1
                while self.size < self.n:
                    self.size <<= 1
                self.minf = float('-inf')
                self.data = [self.minf] * (2 * self.size)
                # Fill the leaves with basket capacities
                for i in range(self.n):
                    self.data[self.size + i] = data[i]
                # Fill the remaining leaves with -infinity (padding)
                for i in range(self.n, self.size):
                    self.data[self.size + i] = self.minf
                # Build the segment tree from the leaves up
                for i in range(self.size - 1, 0, -1):
                    self.data[i] = max(self.data[2 * i], self.data[2 * i + 1])
            
            def update(self, index, value):
                # Update the position in the segment tree
                pos = self.size + index
                self.data[pos] = value
                pos >>= 1  # Move to parent node
                while pos >= 1:
                    new_val = max(self.data[2 * pos], self.data[2 * pos + 1])
                    if self.data[pos] == new_val:
                        break
                    self.data[pos] = new_val
                    pos >>= 1
            
            def query_leftmost(self, fruit):
                # Find the leftmost basket >= fruit
                pos = 1
                left = 0
                right = self.size - 1
                if self.data[1] < fruit:
                    return -1
                while True:
                    if left == right:
                        return left
                    mid = (left + right) // 2
                    left_child = 2 * pos
                    right_child = 2 * pos + 1
                    if self.data[left_child] >= fruit:
                        pos = left_child
                        right = mid
                    else:
                        if self.data[right_child] >= fruit:
                            pos = right_child
                            left = mid + 1
                        else:
                            return -1
        
        st = SegmentTree(baskets)
        unplaced = 0
        for fruit in fruits:
            idx = st.query_leftmost(fruit)
            if idx == -1:
                unplaced += 1
            else:
                st.update(idx, float('-inf'))
        
        return unplaced