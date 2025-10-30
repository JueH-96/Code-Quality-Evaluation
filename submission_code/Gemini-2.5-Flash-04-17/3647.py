import heapq
from collections import defaultdict
from typing import List

# Fenwick Tree (Binary Indexed Tree) implementation for point updates and prefix sum queries
class FenwickTree:
    def __init__(self, size):
        # size is the number of 0-based indices the FT needs to cover.
        # For difference array on an array of size n, we need indices 0 to n for updates.
        # So size should be n + 1.
        self.size = size
        # Internal tree array is 1-based, size + 1
        # It will support 0-based indices 0..size-1 mapped to 1-based 1..size.
        # So if size = n+1, it supports 0-based indices 0..n mapped to 1-based 1..n+1.
        self.tree = [0] * (self.size + 1)

    def update(self, idx, val):
        # Update value at index idx (0-based, 0..size-1)
        # Here size is n+1, so idx can be 0..n.
        idx += 1 # Convert 0-based index to 1-based (1..size)
        while idx <= self.size:
            self.tree[idx] += val
            idx += idx & (-idx)

    def query(self, idx):
        # Get prefix sum up to index idx (0-based, 0..size-1) inclusive
        # Here size is n+1, so idx can be 0..n.
        # We typically query for indices 0..n-1 in the original array.
        # query(i) for i in 0..n-1 calculates sum(diff[0..i]).
        idx += 1 # Convert 0-based index to 1-based (1..size)
        total = 0
        while idx > 0:
            total += self.tree[idx]
            idx -= idx & (-idx)
        return total

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)

        # Fenwick Tree for difference array.
        # We need to handle updates at 0-based indices 0 to n for difference array.
        # FT size needs to be n + 1 to cover these indices (0-based).
        # FT indices will range from 0 to n.
        ft = FenwickTree(n + 1)

        # Store queries grouped by their right endpoint
        # Value is (l, original_query_index)
        queries_by_r = defaultdict(list)
        for i, (l, r) in enumerate(queries):
            queries_by_r[r].append((l, i))

        # Min-priority queue for candidate queries available.
        # Stores (l, original_query_index).
        # Queries in this PQ at step `i` are those `[l, r]` with `r >= i` that haven't been selected.
        # Priority is by left endpoint l.
        candidate_pq = [] # Stores (l, original_query_index)

        selected_count = 0 # Total number of queries selected by the greedy algorithm

        # Iterate from right to left
        for i in range(n - 1, -1, -1):
            # 1. Add queries ending exactly at index i to candidate_pq
            # These queries become available candidates to cover index i and indices to its left (down to their l).
            if i in queries_by_r:
                for l, query_idx in queries_by_r[i]:
                    heapq.heappush(candidate_pq, (l, query_idx))

            # 2. Calculate current coverage at index i from selected queries.
            # This is the prefix sum up to index i in the difference array stored in the Fenwick Tree.
            # The value ft.query(i) gives sum of diff[0...i]. This is the total coverage at index i.
            current_coverage_at_i = ft.query(i) # Query FT for sum up to index i (0-based)

            # 3. Deficit at index i
            deficit = nums[i] - current_coverage_at_i

            # 4. If deficit > 0, we need to select more queries that cover index i.
            # We select from candidate_pq, prioritizing those with minimum l.
            # We only consider queries from candidate_pq whose left endpoint l is <= i,
            # because queries with l > i do not cover the current index i.
            needed = deficit
            
            # Store updates to apply to FT after determining how many are needed
            ft_updates_in_this_step = [] # Stores (index, value) pairs

            # Keep trying to select queries until deficit is covered
            while needed > 0:
                # Try to find a suitable query in candidate_pq that covers index i (i.e., l <= i)
                
                # Check if there are any candidates left
                if not candidate_pq:
                    # Ran out of candidates before satisfying deficit
                    return -1 # Indicate impossible

                l, query_idx = heapq.heappop(candidate_pq)
                
                if l <= i:
                    # Found a suitable query [l, r] where l <= i and r >= i (since it's in candidate_pq at step i)
                    r = queries[query_idx][1] # Get r
                    
                    # This query is selected
                    ft_updates_in_this_step.append((l, 1))
                    # Update at r+1. This affects coverage at indices > r.
                    # FT size is n+1, covering 0-based indices 0..n. Max r+1 is n.
                    # So update at r+1 is always within FT bounds [0, n].
                    ft_updates_in_this_step.append((r + 1, -1))
                    
                    selected_count += 1
                    needed -= 1
                else:
                     # This query [l, r] has l > i. It cannot cover index i or any index < i.
                     # According to the greedy strategy, we discard this query permanently.
                     # Continue the loop to find another candidate.
                     pass 

            # Apply the updates to the Fenwick Tree for the queries selected in this step
            # These updates reflect the contribution of newly selected queries to the difference array
            # and will affect the coverage calculation for indices < i
            for idx, val in ft_updates_in_this_step:
                ft.update(idx, val)

        # If the loop finishes, it means all requirements nums[i] were met using selected_count queries.
        # The minimum number of queries required is selected_count.
        # The maximum number of queries to remove is m - selected_count.
        return m - selected_count