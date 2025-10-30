import bisect
from typing import List

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        pairs = list(zip(nums1, nums2))
        n = len(pairs)
        sums = [a + b for a, b in pairs]
        
        all_b = [b for a, b in pairs]
        all_y = [y for x, y in queries]
        sorted_coords = sorted(list(set(all_b + all_y)))
        m = len(sorted_coords)
        
        # Prepare sorted pairs
        sorted_pairs = sorted(zip(pairs, sums), key=lambda x: (-x[0][0], -x[0][1]))
        
        # Prepare sorted queries
        sorted_queries = []
        for i, (x, y) in enumerate(queries):
            sorted_queries.append( (-x, -y, i) )  # negative for descending sort
        sorted_queries.sort()
        # Now, convert back to (x, y, i)
        processed_queries = []
        for sq in sorted_queries:
            x = -sq[0]
            y = -sq[1]
            idx = sq[2]
            processed_queries.append( (x, y, idx) )
        
        # Implement Segment Tree
        class SegmentTree:
            def __init__(self, size):
                self.size = 1
                while self.size < size:
                    self.size <<= 1
                self.tree = [ -float('inf') ] * (2 * self.size)
            
            def update(self, pos, value):
                pos += self.size
                if self.tree[pos] < value:
                    self.tree[pos] = value
                    pos >>= 1
                    while pos >= 1:
                        new_val = max( self.tree[2*pos], self.tree[2*pos+1] )
                        if self.tree[pos] == new_val:
                            break
                        self.tree[pos] = new_val
                        pos >>= 1
            
            def query_range(self, l, r):
                res = -float('inf')
                l += self.size
                r += self.size
                while l <= r:
                    if l % 2 == 1:
                        res = max(res, self.tree[l])
                        l += 1
                    if r % 2 == 0:
                        res = max(res, self.tree[r])
                        r -= 1
                    l >>= 1
                    r >>= 1
                return res if res != -float('inf') else -1
        
        st = SegmentTree(m)
        answer = [-1] * len(queries)
        ptr = 0  # pointer for sorted_pairs
        
        for x, y, idx in processed_queries:
            # Add all pairs with a >= x
            while ptr < n:
                (a, b), s = sorted_pairs[ptr]
                if a >= x:
                    # find position in sorted_coords
                    pos = bisect.bisect_left(sorted_coords, b)
                    st.update(pos, s)
                    ptr += 1
                else:
                    break
            # Now query for y
            pos_y = bisect.bisect_left(sorted_coords, y)
            if pos_y >= m:
                answer[idx] = -1
            else:
                max_sum = st.query_range(pos_y, m-1)
                answer[idx] = max_sum
        return answer