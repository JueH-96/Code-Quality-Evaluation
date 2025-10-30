import heapq
from typing import List

# Fenwick tree implementation (BIT)
class FenwickTree:
    """ Binary Indexed Tree (Fenwick Tree) for prefix sums and point updates. """
    def __init__(self, size):
        """ Initializes the Fenwick tree of a given size.
        Size should be max_index + 1 to handle indices 0 to max_index. """
        # The tree size needs to be size + 1 because BIT uses 1-based indexing internally.
        self.tree = [0] * (size + 1) 

    def update(self, i, delta):
        """ Adds delta to the element at index i. """
        # Convert to 1-based index for BIT operations
        i += 1 
        # Update the tree by adding delta to all relevant nodes
        while i < len(self.tree):
            self.tree[i] += delta
            # Move to the next index that includes i in its range
            i += i & (-i) 

    def query(self, i):
        """ Queries the prefix sum up to index i (inclusive). """
        # Convert to 1-based index
        i += 1 
        s = 0
        # Sum values from relevant nodes up to index i
        while i > 0:
            s += self.tree[i]
            # Move to the parent index in the BIT structure
            i -= i & (-i) 
        return s

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        """
        Calculates the maximum number of queries that can be removed such that
        the remaining queries can still convert nums to a zero array. Uses a sweep-line
        greedy approach with a Fenwick tree and a max heap.

        Args:
            nums: The initial integer array.
            queries: A list of queries, where each query is [l_i, r_i].

        Returns:
            The maximum number of queries that can be removed, or -1 if it's impossible
            to convert nums to a zero array even with all queries.
        """
        n = len(nums)
        m = len(queries)

        # Compute the difference array `d` for `nums`. Size is n+1.
        # `d[i] = nums[i] - nums[i-1]` for i>0, `d[0]=nums[0]`. `d[n] = -nums[n-1]`.
        # The prefix sums `P_d[i] = sum(d[k] for k=0..i)` satisfy `P_d[i] = nums[i]` for i<n, and `P_d[n]=0`.
        d = [0] * (n + 1)
        if n > 0: # Check needed as nums can be technically empty per Python types but constraints say 1 <= n
            d[0] = nums[0]
            for i in range(1, n):
                d[i] = nums[i] - nums[i-1]
            # Ensures that the total sum of d is 0, necessary for consistency checks.
            d[n] = -nums[n-1]

        # Initial feasibility check: Determine if nums can be zeroed using ALL queries.
        # This check verifies if the condition `P_diff_all[i] >= P_d[i]` holds for all i=0..n,
        # where `P_diff_all` is the prefix sum array corresponding to applying all queries.
        bit_all = FenwickTree(n + 1) # Use n+1 size for indices 0..n
        for l, r in queries:
            bit_all.update(l, 1) # Increment count for indices >= l
            if r + 1 <= n:       # Decrement count for indices >= r+1
                bit_all.update(r + 1, -1)
        
        # Check condition P_diff_all[i] >= nums[i] for i=0..n-1
        for i in range(n):
            # `bit_all.query(i)` computes P_diff_all[i]
            coverage = bit_all.query(i) 
            # Compare with P_d[i] which is nums[i]
            if coverage < nums[i]:
                 # If condition fails for any i, it's impossible.
                 return -1
        
        # Boundary condition check P_diff_all[n] >= P_d[n]. P_d[n]=0.
        # P_diff_all[n] should be 0 because each query adds +1 and -1 to the difference array.
        if bit_all.query(n) < 0:
             # Safety check: if sum is not 0 something is wrong or possibly interpretation issue.
             return -1


        # Sweep-line greedy algorithm to find the minimum required queries `k`
        # Initialize BIT for tracking prefix sums of `delta = current_diff - d`.
        # Initially, `current_diff` is 0, so `delta = -d`.
        # We want to apply minimum queries to make `P_delta[i] >= 0` for all `i`.
        bit = FenwickTree(n + 1)
        for i in range(n + 1):
            if d[i] != 0:
                 # Initialize BIT with the effect of -d
                 bit.update(i, -d[i])

        # Group queries by their start index 'l' for efficient activation during the sweep line
        queries_by_start = [[] for _ in range(n)]
        for idx, q in enumerate(queries):
             l, r = q
             if l < n: # Consider only queries starting within the valid range [0, n-1]
                 queries_by_start[l].append( (r, idx) ) # Store (end_index, query_original_index)

        # Max heap stores tuples (-r, l, idx) to prioritize queries with the largest 'r' endpoint.
        # Using negative 'r' because heapq implements a min-heap.
        max_heap = [] 
        # Keep track of indices of queries selected by the greedy algorithm
        used_query_indices = set() 
        count_selected_queries = 0 # Counts the minimum number of queries needed (k)

        # Sweep line process across indices i = 0 to n-1
        for i in range(n):
            # Activate queries starting at index i by adding them to the max heap
            for r, idx in queries_by_start[i]:
                 heapq.heappush(max_heap, (-r, i, idx)) # Push (-r, start_index, query_index)

            # Check the prefix sum condition P_delta[i] >= 0
            # `bit.query(i)` calculates the current P_delta[i]
            while bit.query(i) < 0:
                 # If P_delta[i] < 0, it means P_current_diff[i] < P_d[i], requiring more coverage at index i.
                 
                 # Clean up the top of the heap: Remove invalid entries.
                 # Invalid entries are queries already used or queries that end before the current index i (r < i).
                 while max_heap:
                      neg_r, l, idx = max_heap[0] # Peek at the top element
                      r = -neg_r
                      if idx in used_query_indices or r < i:
                          # This query is invalid for the current step, remove it from the heap.
                          heapq.heappop(max_heap)
                      else:
                          # The top element is a valid candidate query.
                          break 
                 
                 # If heap becomes empty after cleanup, no available query can cover index i.
                 if not max_heap:
                     # This scenario implies impossibility, which should ideally be caught by the initial feasibility check.
                     # Returning -1 as a defensive measure.
                     return -1 

                 # Pop the best query according to the greedy strategy (covers i, has the largest r).
                 neg_r, l, idx = heapq.heappop(max_heap)
                 r = -neg_r
                 
                 # Apply this selected query: Update the delta array via BIT.
                 # This increases P_delta[k] by 1 for all k in the range [l, r].
                 bit.update(l, 1) # Increment delta at start index l
                 if r + 1 <= n:    # Decrement delta at end index r+1
                     bit.update(r + 1, -1)
                 
                 # Record the query as used and increment the count of selected queries.
                 used_query_indices.add(idx)
                 count_selected_queries += 1

        # The greedy algorithm finds the minimum number of queries `k = count_selected_queries` required.
        # The maximum number of queries that can be removed is `m - k`.
        return m - count_selected_queries