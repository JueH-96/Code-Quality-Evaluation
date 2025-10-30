from typing import List
import heapq

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        
        # Annotate queries with their l and r, sort by l
        intervals = sorted(queries, key=lambda x: x[0])
        
        # We will sweep from i = 0..n-1, maintaining:
        # 1) cover_diff: difference array for how many intervals chosen
        #    start/end at each position
        # 2) a max-heap of candidate intervals (by farthest right endpoint)
        cover_diff = [0] * (n + 1)
        heap = []  # max-heap of (-r, r_index) for intervals with l <= i
        idx = 0    # pointer into sorted intervals
        
        covered = 0   # current coverage at position i
        used = 0      # number of intervals we have chosen
        
        for i in range(n):
            # update coverage by processing diff
            covered += cover_diff[i]
            
            # add all intervals whose l <= i to the heap
            while idx < m and intervals[idx][0] <= i:
                l, r = intervals[idx]
                # push by negative r to make it a max-heap
                heapq.heappush(heap, (-r, r))
                idx += 1
            
            # demand at i
            need = nums[i]
            # while we have less coverage than demand, pick an interval
            while covered < need:
                if not heap:
                    # no interval can cover position i enough times
                    return -1
                # pick the interval with the farthest r
                neg_r, r = heapq.heappop(heap)
                # choose it: it contributes +1 to cover_diff[i..r]
                covered += 1
                used += 1
                # we start its effect at i, end at r+1
                # but we've already added to covered for i
                cover_diff[r+1] -= 1  # when we move past r, we lose its effect
            
        # minimal intervals used = used
        # max removable = total - used
        return m - used