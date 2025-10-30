from typing import List
import heapq

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        # The idea:
        # Each query is an interval [l, r] which, when used, gives +1 “coverage”
        # (or decrement effect) to every index in [l, r].
        # To reduce nums to a zero array we need, for each index i,
        # the sum of all chosen queries covering i to be at least nums[i].
        #
        # We want to remove as many queries as possible, i.e. choose a smallest
        # possible subset of queries so that for every index i, the number of queries
        # chosen that cover i is at least nums[i].
        #
        # We solve this “covering with multiplicities” problem greedily.
        # We iterate index i from 0 to n - 1. Let "added" be the total contributions
        # from queries we have already chosen that cover i. If added < nums[i],
        # we need to choose additional queries that cover i.
        #
        # Which queries to choose? We can choose from those queries that become “available”
        # at i (i.e. queries with l <= i) and have i <= r (they cover i). To minimize the total
        # count, we greedily choose queries that extend as far right as possible so that they can
        # help cover future indices too.
        #
        # We use a max-heap (implemented with negatives of r) to get the query among those available
        # with the maximum r. When we pick a query [l,r] at index i, we know it adds +1 to each index from
        # i to r. We simulate that effect with a difference array.
        
        n = len(nums)
        m = len(queries)
        queries.sort(key=lambda x: x[0])
        
        # diff array is used to simulate the contribution of chosen queries over indices.
        diff = [0]*(n+1)
        added = 0  # cumulative coverage at the current index from chosen queries.
        chosen_count = 0  # number of queries we must keep.
        
        # A max-heap for queries available at the current index.
        # We push each available query as (-r, r) so that the query with the largest r comes first.
        available = []
        q_ptr = 0  # pointer to iterate over sorted queries
        
        # Process each index
        for i in range(n):
            # Update the coverage at index i from our chosen queries.
            added += diff[i]
            
            # Add all queries starting at or before i into the available heap.
            while q_ptr < m and queries[q_ptr][0] <= i:
                l, r = queries[q_ptr]
                if r >= i:  # only add if the query actually covers i
                    heapq.heappush(available, (-r, r))
                q_ptr += 1
            
            # Remove queries that have expired (their r is less than i)
            while available and available[0][1] < i:
                heapq.heappop(available)
            
            # If current coverage is not enough, select additional queries.
            while added < nums[i]:
                if not available:
                    # There is no available query to cover the deficit at i.
                    return -1
                
                # Choose the available query with the furthest r.
                neg_r, r = heapq.heappop(available)
                chosen_count += 1
                # Using this query instantly adds +1 to index i (and will add to indices i+1..r).
                added += 1
                # We update diff so that indices r+1 will lose this added coverage.
                if r + 1 < len(diff):
                    diff[r+1] -= 1
        
        # If the entire array is covered, the minimal set of queries kept is 'chosen_count'.
        # Therefore, maximum removals = total queries - chosen_count.
        return m - chosen_count

# For local testing:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.maxRemoval([2, 0, 2], [[0, 2], [0, 2], [1, 1]]))  # Expected output: 1
    # Example 2:
    print(sol.maxRemoval([1, 1, 1, 1], [[1, 3], [0, 2], [1, 3], [1, 2]]))  # Expected output: 2
    # Example 3:
    print(sol.maxRemoval([1, 2, 3, 4], [[0, 3]]))  # Expected output: -1