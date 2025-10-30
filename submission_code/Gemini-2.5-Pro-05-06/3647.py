import collections
import heapq

class Solution:
  def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
    n = len(nums)
    m = len(queries)

    # Store queries indexed by their start points
    # queries_by_start[i] will be a list of end points r for queries starting at i
    queries_by_start = collections.defaultdict(list)
    for i in range(m):
      l, r = queries[i]
      # According to constraints: 0 <= l_i <= r_i < nums.length
      queries_by_start[l].append(r)

    # Min-heap for active chosen queries, stores their end points r.
    # These are queries whose intervals [l_q, r_q] satisfy l_q <= j <= r_q,
    # and they have been "chosen" to provide coverage.
    active_chosen_queries_heap = []  # Stores r values directly
    
    # Max-heap for available queries that have started (l_q <= j) but not yet chosen.
    # Stores their end points r (as -r to use Python's min-heap as a max-heap).
    # Note: they might not cover current j if r_q < j. This is handled when popping / cleaning up.
    available_not_chosen_heap = [] # Stores -r values

    num_queries_chosen = 0

    for j in range(n): # Iterate through each index of nums from 0 to n-1
      # Add queries starting at current index j to the available_not_chosen_heap.
      # These queries become candidates to be chosen.
      if j in queries_by_start:
        for r_val in queries_by_start[j]:
          heapq.heappush(available_not_chosen_heap, -r_val) # Store -r for max-heap behavior
      
      # Remove queries from active_chosen_queries_heap that have ended before current index j.
      # An active query (represented by its r_q in heap) ends before j if r_q < j.
      while active_chosen_queries_heap and active_chosen_queries_heap[0] < j:
        heapq.heappop(active_chosen_queries_heap)

      # We need nums[j] "coverage" (i.e., nums[j] active chosen queries covering j).
      # Current coverage at index j is len(active_chosen_queries_heap).
      # If coverage is insufficient, we must pick more queries.
      # Greedily pick from available_not_chosen_heap those with the largest r_val (furthest end point).
      while len(active_chosen_queries_heap) < nums[j]:
        # Before picking, clean up available_not_chosen_heap:
        # Remove queries from available_not_chosen_heap if their r_val < j, 
        # as they are expired and useless for current j (and any subsequent k > j).
        while available_not_chosen_heap and -available_not_chosen_heap[0] < j:
            heapq.heappop(available_not_chosen_heap) # Pop -r_val, actual r_val is -(-r_val)

        if not available_not_chosen_heap:
          # Not enough queries to cover nums[j] even after considering all relevant available ones.
          # This means it's impossible to make nums a zero array with this strategy.
          # Since the strategy is optimal for minimizing chosen queries, it's impossible overall.
          return -1
        
        # Get query with largest r_val from available_not_chosen_heap.
        # (It's stored as -r, so it's the smallest element in this min-heap of negative r_vals)
        # Its r_val is guaranteed to be >= j due to the cleanup loop above.
        r_of_best_available = -heapq.heappop(available_not_chosen_heap)
        
        # "Choose" this query: add its r_val to active_chosen_queries_heap
        heapq.heappush(active_chosen_queries_heap, r_of_best_available)
        num_queries_chosen += 1
    
    # If we successfully provided sufficient coverage for all nums[j],
    # num_queries_chosen is the minimum number of queries required.
    # We want to maximize removed queries, which is m - num_queries_chosen.
    return m - num_queries_chosen