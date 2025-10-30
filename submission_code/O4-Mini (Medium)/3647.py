import heapq
from typing import List

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        # Compute total coverage of each position
        diff = [0] * (n + 1)
        for l, r in queries:
            diff[l] += 1
            if r + 1 <= n - 1:
                diff[r + 1] -= 1
        cover = [0] * n
        cur = 0
        for i in range(n):
            cur += diff[i]
            cover[i] = cur
        # Compute capacity: how many queries we may remove covering i
        cap = [cover[i] - nums[i] for i in range(n)]
        for c in cap:
            if c < 0:
                return -1

        # Build list of queries starting at each position
        starts = [[] for _ in range(n)]
        for idx, (l, r) in enumerate(queries):
            starts[l].append((r, idx))

        # Heaps to maintain active (removed) intervals
        # active_maxheap for greedy removal: max by end => store (-r, idx)
        # active_minheap for expiring intervals: min by end => store (r, idx)
        active_maxheap = []
        active_minheap = []
        active = [False] * m  # whether interval idx is currently in the removed set R and covers current t

        active_count = 0    # number of removed intervals covering current t
        total_removed = 0   # total intervals removed in R

        # Sweep from t = 0 to n-1
        for t in range(n):
            # 1) Add new intervals starting at t into R (optimistically)
            for r, idx in starts[t]:
                active[idx] = True
                active_count += 1
                total_removed += 1
                heapq.heappush(active_maxheap, (-r, idx))
                heapq.heappush(active_minheap, (r, idx))

            # 2) Expire intervals whose r < t (they no longer cover position t)
            while active_minheap and active_minheap[0][0] < t:
                r0, idx0 = heapq.heappop(active_minheap)
                if active[idx0]:
                    # it was counted in active_count, now drop it
                    active[idx0] = False
                    active_count -= 1

            # 3) If over capacity at t, remove intervals greedily
            #    (remove those with furthest end first)
            while active_count > cap[t]:
                # pop from max-heap until we find an actually active interval
                while active_maxheap:
                    neg_r, idx2 = heapq.heappop(active_maxheap)
                    if active[idx2]:
                        break
                # idx2 is to be removed from R
                active[idx2] = False
                active_count -= 1
                total_removed -= 1
                # Note: its entry remains in active_minheap but will be ignored later

        return total_removed