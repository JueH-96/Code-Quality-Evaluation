class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def add(self, idx, val):
        while idx <= self.n:
            self.tree[idx] += val
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

    def range_add(self, l, r, val):
        # Converts 0-based indices to 1-based for the Fenwick Tree
        l += 1
        r += 1
        self.add(l, val)
        self.add(r + 1, -val)

    def point_query(self, i):
        # Converts 0-based index to 1-based for the query
        i += 1
        return self.query(i) - self.query(i - 1)

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        # Sort queries by their left endpoint while keeping track of original indices
        sorted_queries = sorted(enumerate(queries), key=lambda x: x[1][0])
        ft = FenwickTree(n)
        heap = []
        used = [False] * m
        selected = 0
        j = 0  # Pointer for traversing sorted_queries
        
        for i in range(n):
            # Add all queries with left endpoint <= i
            while j < m and sorted_queries[j][1][0] <= i:
                original_idx, (l, r) = sorted_queries[j]
                heapq.heappush(heap, (-r, original_idx))
                j += 1
            
            # Remove queries whose right endpoint is < i
            while heap and -heap[0][0] < i:
                heapq.heappop(heap)
            
            # Check if current coverage meets the requirement
            current = ft.point_query(i)
            needed = nums[i] - current
            if needed > 0:
                count = 0
                while heap and count < needed:
                    _, original_idx = heapq.heappop(heap)
                    if not used[original_idx]:
                        used[original_idx] = True
                        selected += 1
                        l, r = queries[original_idx]
                        ft.range_add(l, r, 1)
                        count += 1
                if count < needed:
                    return -1
        
        return m - selected