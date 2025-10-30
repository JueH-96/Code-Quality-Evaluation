class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def update(self, idx, delta):
        idx += 1  # Convert to 1-based index
        while idx <= self.n + 1:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        idx += 1  # Convert to 1-based index
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

    def range_update(self, l, r, val):
        self.update(l, val)
        self.update(r + 1, -val)

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        
        # Step 1: Check if initial coverage is possible
        diff = [0] * (n + 2)  # Handles up to r+1 = n+1
        for l, r in queries:
            diff[l] += 1
            if r + 1 <= n:
                diff[r + 1] -= 1
        
        count_cover = [0] * n
        current = 0
        possible = True
        for i in range(n):
            current += diff[i]
            count_cover[i] = current
            if current < nums[i]:
                possible = False
        if not possible:
            return -1
        
        # Step 2: Sort queries by left endpoint
        queries.sort(key=lambda x: x[0])
        
        # Step 3: Initialize Fenwick Tree and priority queue
        ft = FenwickTree(n)
        heap = []
        idx = 0
        count_selected = 0
        
        # Step 4: Process each position
        for i in range(n):
            # Add applicable queries to the heap
            while idx < m and queries[idx][0] <= i:
                l, r = queries[idx]
                if r >= i:
                    heapq.heappush(heap, (-r, l, r))
                idx += 1
            
            # Remove queries from heap with r < i
            while heap and (-heap[0][0]) < i:
                heapq.heappop(heap)
            
            # Check coverage and select queries if needed
            current_coverage = ft.query(i)
            needed = nums[i]
            deficit = needed - current_coverage
            if deficit < 0:
                deficit = 0
            
            while deficit > 0 and heap:
                _, l, r = heapq.heappop(heap)
                ft.range_update(l, r, 1)
                count_selected += 1
                deficit -= 1
            
            if deficit > 0:
                return -1
        
        return m - count_selected