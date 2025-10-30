import collections
import heapq
from typing import List

class SegmentTree:
    def __init__(self, arr: List[int], N: int):
        self.N = N
        # tree stores the minimum value in the segment
        self.tree = [0] * (4 * N) 
        # lazy stores pending updates for the segment
        self.lazy = [0] * (4 * N)
        self._build(arr, 1, 0, N - 1)

    def _build(self, arr: List[int], node: int, start: int, end: int):
        """Recursively builds the segment tree."""
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def _push_down(self, node: int):
        """Applies lazy updates to children and clears lazy tag."""
        if self.lazy[node] != 0:
            # Apply lazy tag to children's tree values
            self.tree[2 * node] += self.lazy[node]
            self.tree[2 * node + 1] += self.lazy[node]
            # Pass lazy tag to children
            self.lazy[2 * node] += self.lazy[node]
            self.lazy[2 * node + 1] += self.lazy[node]
            # Clear current node's lazy tag
            self.lazy[node] = 0

    def range_add(self, query_l: int, query_r: int, val: int):
        """Adds `val` to elements in range [query_l, query_r]."""
        self._range_add(1, 0, self.N - 1, query_l, query_r, val)

    def _range_add(self, node: int, start: int, end: int, query_l: int, query_r: int, val: int):
        """Recursive helper for range_add."""
        # No overlap
        if query_r < start or end < query_l: 
            return
        # Complete overlap
        if query_l <= start and end <= query_r: 
            self.tree[node] += val
            self.lazy[node] += val
            return

        # Partial overlap: push down lazy tag, then recurse
        self._push_down(node)
        mid = (start + end) // 2
        self._range_add(2 * node, start, mid, query_l, query_r, val)
        self._range_add(2 * node + 1, mid + 1, end, query_l, query_r, val)
        # Update current node's value from children
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def query_min(self, query_l: int, query_r: int) -> int:
        """Queries the minimum value in range [query_l, query_r]."""
        return self._query_min(1, 0, self.N - 1, query_l, query_r)

    def _query_min(self, node: int, start: int, end: int, query_l: int, query_r: int) -> int:
        """Recursive helper for query_min."""
        # No overlap, return a value that won't affect min calculation
        if query_r < start or end < query_l: 
            return float('inf')
        # Complete overlap
        if query_l <= start and end <= query_r: 
            return self.tree[node]

        # Partial overlap: push down lazy tag, then recurse
        self._push_down(node)
        mid = (start + end) // 2
        p1 = self._query_min(2 * node, start, mid, query_l, query_r)
        p2 = self._query_min(2 * node + 1, mid + 1, end, query_l, query_r)
        return min(p1, p2)


class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        N = len(nums)
        Q = len(queries)

        def check(k_limit: int) -> bool:
            """
            Checks if it's possible to convert `nums` to a zero array
            using at most `k_limit` queries.
            """
            seg_tree = SegmentTree(nums, N)
            
            # Group queries by their start index for efficient retrieval
            # queries_by_start[l] will be a list of (r, original_query_idx) for queries starting at l
            queries_by_start = collections.defaultdict(list)
            for q_idx, (l, r) in enumerate(queries):
                queries_by_start[l].append((r, q_idx))

            # Max-heap to store active queries, ordered by their right endpoint (descending)
            # We store (-r, q_idx) to simulate a max-heap with Python's min-heap `heapq`
            max_heap_active_queries = [] 
            
            num_queries_used = 0

            # Iterate through each index of the nums array
            for i in range(N):
                # Add all queries that start at the current index `i` to the heap
                if i in queries_by_start:
                    for r, q_idx in queries_by_start[i]:
                        heapq.heappush(max_heap_active_queries, (-r, q_idx))

                # Remove queries from the heap that have already ended (their right endpoint `r` is less than `i`)
                # These queries cannot cover the current index `i` or any subsequent indices.
                while max_heap_active_queries and -max_heap_active_queries[0][0] < i:
                    heapq.heappop(max_heap_active_queries)
                
                # Get the current value needed for index `i` from the segment tree.
                # This value accounts for all range decrements applied by previously selected queries.
                needed_at_i = seg_tree.query_min(i, i)
                
                # While index `i` still needs decrements (i.e., its value in seg_tree is > 0)
                while needed_at_i > 0:
                    # If no active queries can cover `i` (heap is empty), then it's impossible.
                    if not max_heap_active_queries:
                        return False
                    
                    # Pop the query with the largest right endpoint from the heap.
                    # This greedy choice maximizes the potential future utility of the selected query.
                    r_chosen, q_idx_chosen = heapq.heappop(max_heap_active_queries)
                    r_chosen = -r_chosen # Convert back to original r value
                    
                    # This check is a safeguard; a query popped here should always cover `i` due to the
                    # purging loop above. If it somehow doesn't, it implies an issue or an edge case.
                    if r_chosen < i:
                        # If the chosen query does not cover 'i', it means it expired.
                        # We should continue to find another query from the heap that *does* cover 'i'.
                        # The complexity analysis implicitly assumes queries are always useful when popped here.
                        # This `continue` ensures correctness, even if it slightly perturbs worst-case `logQ` per pop.
                        continue

                    num_queries_used += 1
                    # If we have used more queries than our allowed limit `k_limit`, then this `k_limit` is not sufficient.
                    if num_queries_used > k_limit:
                        return False
                    
                    # Apply the decrement from the chosen query to the range [i, r_chosen] in the segment tree.
                    # We decrement starting from `i` because this query is specifically chosen to satisfy `i`
                    # and any prior contributions for indices < `i` are already reflected in `needed_at_i`.
                    seg_tree.range_add(i, r_chosen, -1)
                    
                    # Re-evaluate the need at index `i` after applying the decrement.
                    needed_at_i = seg_tree.query_min(i, i)
            
            # After iterating through all indices, check if all elements in `nums` have become zero or less.
            # This means the minimum value in the entire segment tree should be less than or equal to 0.
            return seg_tree.query_min(0, N - 1) <= 0

        # Binary search for the minimum number of queries to KEEP (`min_kept`)
        low = 0
        high = Q # Maximum possible queries to keep is all of them
        min_kept = -1

        while low <= high:
            mid = low + (high - low) // 2
            if check(mid):
                min_kept = mid
                high = mid - 1 # Try to achieve with fewer queries
            else:
                low = mid + 1 # Need to keep more queries
        
        # If min_kept is -1, it means it's not possible to zero out nums even with all queries.
        if min_kept == -1:
            return -1
        else:
            # The result is the maximum number of queries to REMOVE
            return Q - min_kept