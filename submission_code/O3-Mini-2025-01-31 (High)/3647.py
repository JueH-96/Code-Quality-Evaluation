from typing import List
import heapq

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        
        # ─── STEP 1: Global Feasibility Check ─────────────────────────────
        # Even if we use all queries, each query “offers” a chance to subtract 1 at
        # every index in its interval. Therefore the maximum decrement available at
        # an index is the number of queries covering that index. If for some index
        # this is less than nums[i], it is impossible to make it zero.
        cap = [0] * (n+1)
        for l, r in queries:
            cap[l] += 1
            if r+1 < len(cap):
                cap[r+1] -= 1
        current = 0
        for i in range(n):
            current += cap[i]
            if current < nums[i]:
                return -1  # Not feasible even using all queries.
                
        # ─── STEP 2: Greedy Selection of a Minimum Subset of Queries ─────
        # We want to select a subset of queries so that for every index i the sum of
        # chosen queries that cover i (each contributes 1) is at least nums[i].
        # Equivalently, we want to “cover” each position i with at least nums[i] intervals.
        #
        # Notice that if we can cover all positions, then we can remove the other queries.
        # Our answer will be:
        #      answer = total queries m  - (minimum queries needed)
        #
        # Because every query is an independent unit (can subtract at most 1 per index),
        # we can solve the following “covering” problem:
        #   For i = 0 to n-1, we want the cumulative sum of contributions to be ≥ nums[i].
        #
        # A greedy “sweep‐line” procedure is applicable for interval covering if we
        # “fix” a deficit at the current index by choosing an interval (query) among those
        # available that extends as far as possible.
        
        # Sort queries by their starting index.
        sorted_queries = sorted(queries, key=lambda x: x[0])
        # diff[] will simulate the effect of chosen queries.
        # When we choose an interval query covering [l, r], we add +1 to diff[l] and -1 to diff[r+1].
        diff = [0] * (n+1)
        active = 0    # active extra coverage at the current index (from queries already chosen)
        chosen = 0    # number of chosen queries (i.e. minimum queries needed)
        
        # We will use a max‐heap (simulated with a min‐heap using negative values) for candidate queries.
        # At any index i, we want to be able to quickly pick a query with l <= i that covers i and extends far to the right.
        heap = []
        ptr = 0     # pointer to iterate over sorted_queries
        
        for i in range(n):
            # Update the current additional coverage at i using the diff array.
            active += diff[i]
            
            # Add all queries that start at or before i – they can potentially help for index i.
            while ptr < m and sorted_queries[ptr][0] <= i:
                l, r = sorted_queries[ptr]
                if r >= i:  # Only intervals that actually cover i are useful.
                    # Push using key -r. The candidate with the smallest -r has the largest r.
                    heapq.heappush(heap, (-r, r))
                ptr += 1
            
            # If the current coverage is insufficient at i, we must “fix” the deficit.
            if active < nums[i]:
                deficit = nums[i] - active
                for _ in range(deficit):
                    # Before choosing a query, make sure that the candidate on top can cover i.
                    while heap and heap[0][1] < i:
                        heapq.heappop(heap)
                    if not heap:
                        return -1  # Should not happen because global feasibility was checked.
                    # Choose the query with the farthest r (largest coverage to the right).
                    _, r = heapq.heappop(heap)
                    chosen += 1
                    # Immediately add its effect at index i.
                    active += 1
                    # Mark that its coverage will “disappear” after index r.
                    if r+1 < len(diff):
                        diff[r+1] -= 1
        
        # Maximum queries removable equals total queries minus the minimum queries needed.
        return m - chosen

# ─── Example Test Cases ──────────────────────────────────────────────
if __name__ == '__main__':
    sol = Solution()
    # Example 1: nums = [2,0,2], queries = [[0,2],[0,2],[1,1]] -> Output: 1
    print(sol.maxRemoval([2, 0, 2], [[0, 2], [0, 2], [1, 1]]))
    # Example 2: nums = [1,1,1,1], queries = [[1,3],[0,2],[1,3],[1,2]] -> Output: 2
    print(sol.maxRemoval([1, 1, 1, 1], [[1, 3], [0, 2], [1, 3], [1, 2]]))
    # Example 3: nums = [1,2,3,4], queries = [[0,3]] -> Output: -1
    print(sol.maxRemoval([1, 2, 3, 4], [[0, 3]]))