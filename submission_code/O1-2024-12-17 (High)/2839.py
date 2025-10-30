class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        import sys
        import bisect
        
        n = len(nums1)
        q = len(queries)
        
        # Combine nums1, nums2 into points: (x, y, sum)
        points = []
        for i in range(n):
            points.append((nums1[i], nums2[i], nums1[i] + nums2[i]))
        
        # Prepare queries in the form (x, y, original_index)
        # so we can return results in correct order
        queries_with_index = []
        for i, (x, y) in enumerate(queries):
            queries_with_index.append((x, y, i))
        
        # Coordinate compress all y-values (from both points and queries)
        all_y = set()
        for _, y, _ in queries_with_index:
            all_y.add(y)
        for _, y, _ in points:
            all_y.add(y)
        sorted_y = sorted(all_y)
        size_y = len(sorted_y)
        
        # Reversed-rank function: 
        # we want to map yVal -> an index in Fenwick that corresponds to "all y >= yVal"
        # i = bisect_left(sorted_y, yVal) gives index where yVal could be inserted (0-based)
        # reversed_rank = size_y - i
        # if i == size_y => means no larger or equal y, return 0 to indicate invalid
        def get_reversed_rank(yVal):
            i = bisect.bisect_left(sorted_y, yVal)
            if i == size_y:
                return 0  # no valid index for y >= yVal
            return size_y - i
        
        # Sort points by x descending, if tie by y descending (to add largest x first)
        points.sort(key=lambda x: (x[0], x[1]), reverse=True)
        # Sort queries by x descending
        queries_with_index.sort(key=lambda x: x[0], reverse=True)
        
        # Fenwick (BIT) for range max
        class Fenwick:
            def __init__(self, n):
                self.n = n
                # store max; initialize to -1
                self.data = [-1] * (n + 1)
            
            def update(self, idx, val):
                while idx <= self.n:
                    if val > self.data[idx]:
                        self.data[idx] = val
                    idx += idx & -idx
            
            def query(self, idx):
                # prefix max from 1..idx
                res = -1
                while idx > 0:
                    if self.data[idx] > res:
                        res = self.data[idx]
                    idx -= idx & -idx
                return res
        
        fenw = Fenwick(size_y)
        ans = [-1] * q
        
        # We'll move through points in descending x, and queries in descending x
        p_idx = 0
        for xQ, yQ, q_idx in queries_with_index:
            # Add all points with x >= xQ to Fenwicks
            while p_idx < n and points[p_idx][0] >= xQ:
                xP, yP, sumP = points[p_idx]
                # Insert into Fenwicks by reversed-rank of yP
                rrank = get_reversed_rank(yP)
                if rrank > 0:
                    fenw.update(rrank, sumP)
                p_idx += 1
            
            # Now answer this query: max among y >= yQ
            rrankQ = get_reversed_rank(yQ)
            if rrankQ == 0:
                ans[q_idx] = -1  # no valid y >= yQ
            else:
                ans[q_idx] = fenw.query(rrankQ)
        
        return ans