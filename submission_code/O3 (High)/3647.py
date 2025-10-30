from typing import List
import heapq

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)

        # ---------- 1.  Feasibility test with all queries ----------
        #    build coverage array using a difference array
        diff = [0]*(n+1)
        for l, r in queries:
            diff[l] += 1
            if r + 1 < n:
                diff[r+1] -= 1

        cov = 0
        for i in range(n):
            cov += diff[i]
            if cov < nums[i]:          # even with all queries we cannot reach zero array
                return -1

        # ---------- 2.  Greedy selection of a minimum subset ----------
        # sort intervals by left end so that we can add them while sweeping
        intervals = sorted(queries, key=lambda x: x[0])

        cand_heap = []          # max-heap (store –right) with all intervals that already started
        active_heap = []        # min-heap with right ends of intervals that are already selected
        ptr = 0                 # next interval that has not been processed yet
        chosen = 0              # number of intervals we finally keep

        for pos in range(n):
            # remove selected intervals that no longer cover current position
            while active_heap and active_heap[0] < pos:
                heapq.heappop(active_heap)

            # add new intervals whose left end <= current position
            while ptr < m and intervals[ptr][0] <= pos:
                heapq.heappush(cand_heap, (-intervals[ptr][1], ptr))
                ptr += 1

            # how many more intervals do we still need for this position?
            need = nums[pos] - len(active_heap)

            # take the intervals with the farthest right end first
            while need > 0:
                # discard candidates that do not cover current position any more
                while cand_heap and -cand_heap[0][0] < pos:
                    heapq.heappop(cand_heap)

                if not cand_heap:          # should not happen because of the earlier check
                    return -1

                neg_r, idx = heapq.heappop(cand_heap)
                r = -neg_r
                heapq.heappush(active_heap, r)   # interval is now selected
                chosen += 1
                need -= 1

        # we kept `chosen` queries, so we can remove the rest
        return m - chosen