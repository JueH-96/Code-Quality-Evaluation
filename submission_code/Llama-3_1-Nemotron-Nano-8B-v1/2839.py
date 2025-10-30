from bisect import bisect_left
from typing import List

class SegmentTree:
    def __init__(self, size):
        self.n = 1
        while self.n < size:
            self.n <<= 1
        self.size = self.n
        self.tree = [-float('inf')] * (2 * self.n)
    
    def update(self, pos, value):
        pos += self.n  # shift to leaf node
        if self.tree[pos] >= value:
            return
        self.tree[pos] = value
        while pos > 1:
            pos >>= 1
            new_val = max(self.tree[2 * pos], self.tree[2 * pos + 1])
            if self.tree[pos] == new_val:
                break
            self.tree[pos] = new_val
    
    def query_range(self, l, r):
        res = -float('inf')
        l += self.n
        r += self.n
        while l <= r:
            if l % 2 == 1:
                res = max(res, self.tree[l])
                l += 1
            if r % 2 == 0:
                res = max(res, self.tree[r])
                r -= 1
            l >>= 1
            r >>= 1
        return res

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        elements = sorted(zip(nums1, nums2), key=lambda x: x[0])
        n = len(elements)
        
        # Coordinate compression for nums2
        unique_nums2 = sorted(set(nums2))
        max_rank = len(unique_nums2) - 1 if unique_nums2 else -1
        
        # Prepare sorted queries with original indices
        sorted_queries = sorted(enumerate(queries), key=lambda x: (-x[1][0], x[0]))
        original_indices = [q[0] for q in sorted_queries]
        sorted_queries = [(q[1][0], q[1][1]) for q in sorted_queries]
        
        # Initialize segment tree
        st_size = len(unique_nums2)
        if st_size == 0:
            return [-1] * len(queries)
        st = SegmentTree(st_size)
        
        ptr = n - 1
        answer = [-1] * len(queries)
        
        for i in range(len(sorted_queries)):
            x, y = sorted_queries[i]
            # Insert elements with nums1 >= x
            while ptr >= 0 and elements[ptr][0] >= x:
                num1, num2 = elements[ptr]
                # Find the rank of num2
                rank = bisect_left(unique_nums2, num2)
                if rank <= max_rank:
                    st.update(rank, num1 + num2)
                ptr -= 1
            
            # Find the rank for y
            if not unique_nums2:
                max_sum = -float('inf')
            else:
                rank = bisect_left(unique_nums2, y)
                if rank > max_rank:
                    max_sum = -float('inf')
                else:
                    max_sum = st.query_range(rank, max_rank)
            
            # Determine the answer
            if max_sum == -float('inf'):
                answer[original_indices[i]] = -1
            else:
                answer[original_indices[i]] = max_sum if max_sum != -float('inf') else -1
        
        return answer