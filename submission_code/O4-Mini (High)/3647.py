class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        import heapq

        n = len(nums)
        m = len(queries)
        # Sort queries by their start index
        queries_sorted = sorted(queries, key=lambda x: x[0])
        idx = 0

        # A max‐heap (implemented with negated values) of right‐ends of intervals
        cand = []
        # A min‐heap of right‐ends of intervals we have selected (to track which
        # selected intervals have expired as we move i forward)
        sel_ends = []
        active = 0      # how many selected intervals currently cover position i
        cnt_sel = 0     # total number of intervals we've chosen

        for i in range(n):
            # Add all intervals whose start ≤ i into the candidate heap
            while idx < m and queries_sorted[idx][0] <= i:
                l, r = queries_sorted[idx]
                heapq.heappush(cand, -r)
                idx += 1

            # Remove from active those selected intervals whose end < i
            while sel_ends and sel_ends[0] < i:
                heapq.heappop(sel_ends)
                active -= 1

            # We need at least nums[i] intervals covering position i
            need = nums[i]
            while active < need:
                # Discard any candidate intervals that cannot cover i
                while cand and -cand[0] < i:
                    heapq.heappop(cand)
                if not cand:
                    # No way to get enough coverage at i
                    return -1
                # Select the interval with the farthest right end
                r_sel = -heapq.heappop(cand)
                heapq.heappush(sel_ends, r_sel)
                active += 1
                cnt_sel += 1

        # We succeeded in covering all demands using cnt_sel intervals.
        # We can remove the rest.
        return m - cnt_sel