import heapq
from typing import List

class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        """
        We need to assign each unit of decrement for each position i (nums[i] times)
        to one of the queries whose interval [l, r] covers i.  Each query j can
        be used at most (r_j - l_j + 1) times in total (one per distinct index).
        This becomes a flow / matching problem on intervals, but can be solved
        greedily: sweep i from 0..n-1, maintain a min-heap of active queries
        (by their right endpoint).  For each position i with demand d = nums[i],
        we greedily use capacity from the active query that ends earliest.
        If at any point we cannot satisfy demand, return False.
        """
        n = len(nums)
        m = len(queries)
        # For each start index, list queries that begin there
        start_buckets = [[] for _ in range(n)]
        for l, r in queries:
            start_buckets[l].append(r)
        
        # min-heap of (r_j, remaining_capacity_j)
        heap = []
        
        for i in range(n):
            # Add all queries that start at i
            for r in start_buckets[i]:
                cap = r - i + 1
                heapq.heappush(heap, [r, cap])
            
            # Remove queries that ended before i
            while heap and heap[0][0] < i:
                heapq.heappop(heap)
            
            # Now satisfy nums[i] units of demand
            demand = nums[i]
            while demand > 0:
                if not heap:
                    return False
                rj, capj = heapq.heappop(heap)
                # allocate as many units as possible from this query
                use = min(demand, capj)
                demand -= use
                capj -= use
                # if this query still has leftover capacity, push it back
                if capj > 0:
                    heapq.heappush(heap, [rj, capj])
            # done i
        return True