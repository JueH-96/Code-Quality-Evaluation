import heapq
from typing import List

class Solution:
  def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
    """
    Solves the problem by first checking feasibility and then using a greedy sweep-line algorithm.
    The problem asks for the maximum number of queries to remove, which is equivalent to
    minimizing the number of queries to keep.
    """
    n = len(nums)
    m = len(queries)

    # 1. Feasibility Check
    # Check if it's possible using all queries. For each index i, the number of
    # queries covering it must be at least nums[i].
    diff = [0] * (n + 1)
    for l, r in queries:
        diff[l] += 1
        if r + 1 <= n:
            diff[r + 1] -= 1
    
    coverage = 0
    for i in range(n):
        coverage += diff[i]
        if coverage < nums[i]:
            return -1

    # 2. Find minimum queries to keep (min_k) using a greedy sweep-line approach.
    starts_at = [[] for _ in range(n)]
    for l, r in queries:
        starts_at[l].append(r)

    # Max-heap for available queries (stores -r to simulate max-heap with min-heap).
    available_queries_maxheap = []
    # Min-heap for active chosen queries, storing their endpoints.
    active_queries_minheap = []
    
    chosen_count = 0
    
    for i in range(n):
        # Deactivate chosen queries that have ended before index i.
        while active_queries_minheap and active_queries_minheap[0] < i:
            heapq.heappop(active_queries_minheap)

        # Add queries starting at index i to the available pool.
        for r in starts_at[i]:
            heapq.heappush(available_queries_maxheap, -r)
        
        # Prune available queries that are no longer useful (end before i),
        # as they cannot cover index i or any subsequent index.
        while available_queries_maxheap and -available_queries_maxheap[0] < i:
            heapq.heappop(available_queries_maxheap)

        # Calculate how many more queries we need to activate for index i.
        current_coverage = len(active_queries_minheap)
        needed = nums[i] - current_coverage
        
        if needed > 0:
            # The initial feasibility check guarantees we have enough available queries.
            # The total coverage at i is current_coverage + len(available_queries_maxheap).
            # The feasibility check ensures this sum is >= nums[i], so len(available_queries_maxheap) >= needed.
            
            chosen_count += needed
            
            # Activate the 'needed' queries that extend the furthest (largest r).
            for _ in range(needed):
                # Pop from max-heap and push to min-heap
                r_neg = heapq.heappop(available_queries_maxheap)
                heapq.heappush(active_queries_minheap, -r_neg)

    # The result is the total number of queries minus the minimum we must keep.
    return m - chosen_count