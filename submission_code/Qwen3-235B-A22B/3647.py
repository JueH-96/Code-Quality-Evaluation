from bisect import bisect_right
import heapq
from typing import List

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def add(self, idx, delta):
        # Convert to 1-based index
        idx += 1
        while idx < len(self.tree):
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        # Convert to 1-based index and get prefix sum
        idx += 1
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        # Sort queries by their left endpoint, then by right in ascending order
        sorted_queries = sorted(queries, key=lambda x: (x[0], x[1]))
        ft = FenwickTree(n)
        heap = []
        j = 0  # Pointer to added queries
        q = len(queries)
        used = 0  # Number of queries used

        for i in range(n):
            # Add all queries with left <= i
            while j < q and sorted_queries[j][0] <= i:
                l, r = sorted_queries[j]
                # Push to heap with -r for max-heap
                heapq.heappush(heap, (-r, l, r))
                j += 1
            # Remove intervals from heap with r < i
            while heap:
                top_r = -heap[0][0]
                if top_r < i:
                    heapq.heappop(heap)
                else:
                    break
            # Calculate needed coverage
            current_coverage = ft.query(i)
            need = nums[i] - current_coverage
            if need < 0:
                # Over-covered, no need to do anything
                pass
            else:
                while need > 0:
                    if not heap:
                        return -1
                    neg_r, l, r = heapq.heappop(heap)
                    # Apply this interval
                    ft.add(l, 1)
                    ft.add(r + 1, -1)
                    used += 1
                    need -= 1
        # If all elements processed, return the number of removed queries
        return q - used