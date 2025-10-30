import bisect

class SegmentTree:
    def __init__(self, size):
        self.n = 1
        while self.n < size:
            self.n <<= 1
        self.size = self.n
        self.tree = [-float('inf')] * (2 * self.n)
    
    def update_point(self, pos, value):
        pos += self.n
        if self.tree[pos] < value:
            self.tree[pos] = value
            pos >>= 1
            while pos >= 1:
                new_val = max(self.tree[2 * pos], self.tree[2 * pos + 1])
                if self.tree[pos] == new_val:
                    break
                self.tree[pos] = new_val
                pos >>= 1
    
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
    def maximumSumQueries(self, nums1, nums2, queries):
        points = [ (a, b, a + b) for a, b in zip(nums1, nums2) ]
        points.sort(key=lambda x: -x[0])  # Sort points in descending order of a
        
        # Collect all b's and y's from queries
        all_values = []
        for b in nums2:
            all_values.append(b)
        for q in queries:
            all_values.append(q[1])
        # Deduplicate and sort
        compressed = sorted(list(set(all_values)))
        rank_dict = {v:i for i, v in enumerate(compressed)}
        m = len(compressed)
        
        # Sort queries by x in descending order, keeping original index
        sorted_queries = sorted(enumerate(queries), key=lambda x: (-x[1][0], -x[1][1]))
        
        # Initialize the segment tree
        st = SegmentTree(m)
        
        # Prepare the answer list
        ans = [-1] * len(queries)
        
        ptr = 0  # Pointer to the points list
        
        for original_index, (x, y) in sorted_queries:
            # Add all points with a >= x
            while ptr < len(points) and points[ptr][0] >= x:
                a_point, b_point, sum_point = points[ptr]
                # Find the rank of b_point
                b_rank = rank_dict[b_point]
                # Update the segment tree
                st.update_point(b_rank, sum_point)
                ptr += 1
            
            # Find the rank for y using bisect
            idx = bisect.bisect_left(compressed, y)
            if idx >= m:
                # No such b exists
                ans[original_index] = -1
            else:
                # Query the range [idx, m-1]
                max_sum = st.query_range(idx, m-1)
                if max_sum == -float('inf'):
                    ans[original_index] = -1
                else:
                    ans[original_index] = max_sum
        
        return ans