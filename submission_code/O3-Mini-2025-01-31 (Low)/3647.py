from typing import List
import heapq

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        # We'll compute the minimum number of queries needed (min_used) such that for every index i,
        # the total decrement provided by these queries covering i is at least nums[i].
        #
        # Each chosen query [l, r] adds 1 decrement to every index i in [l, r]. We can choose which indices to 
        # actually apply the decrement, but if a query is chosen then its "potential" decrement is available for every index in its interval.
        #
        # This is equivalent to: choose a set of queries (intervals) such that for every index i, 
        # the number of chosen intervals that cover i is at least nums[i]. 
        # We need to minimize the number of queries chosen.
        #
        # A greedy, left-to-right sweep algorithm can solve this kind of “covering with demand” problem.
        #
        # We will maintain a difference array 'diff' for the cumulative effect (the number of chosen queries covering each index).
        # Let supply[i] be the sum of diff[0..i] which tells how many decrements we have "allocated" to index i.
        # We go index by index from 0 to n-1. At each index i, if supply < nums[i],
        # we need to choose additional queries that cover i. We choose them from the set of queries 
        # that start at or before i and not used already (using a max-heap keyed by their ending index).
        #
        # For each needed query, we choose the one with the furthest right r.
        #
        # If at any point we cannot cover the deficit (i.e. no query available), we return -1.
        # Finally, the maximum number of queries that can be removed is total queries minus 
        # the minimum number required.
        
        # Sort queries by their left endpoint.
        sorted_queries = sorted(queries, key=lambda x: x[0])
        # Use a max-heap (simulate with min-heap using negative values) for candidate intervals based on r.
        candidate_heap = []
        used = 0  # number of queries chosen (minimum needed)
        
        diff = [0]*(n+1)  # difference array to update coverage from chosen queries.
        supply = 0        # current coverage at index i
        
        q_idx = 0  # pointer to sorted_queries
        
        for i in range(n):
            # Update supply from diff array.
            supply += diff[i]
            # Add all queries starting at or before i into candidate_heap.
            while q_idx < m and sorted_queries[q_idx][0] <= i:
                # use negative r to simulate max-heap by r
                heapq.heappush(candidate_heap, -sorted_queries[q_idx][1])
                q_idx += 1
            # Remove candidates that cannot cover index i.
            while candidate_heap and -candidate_heap[0] < i:
                heapq.heappop(candidate_heap)
            # If the current supply is less than nums[i], we need to choose extra queries.
            if supply < nums[i]:
                deficit = nums[i] - supply
                for _ in range(deficit):
                    if not candidate_heap:
                        return -1  # cannot cover the required decrement at index i
                    # Choose the candidate with the furthest right endpoint.
                    r = -heapq.heappop(candidate_heap)
                    # When choosing a query [l, r], we can add 1 to every index in [i, r]
                    diff[i] += 1
                    if r+1 < len(diff):
                        diff[r+1] -= 1
                    supply += 1  # we've increased supply at current index by 1
                    used += 1
                    
        # The maximum number of queries that can be removed is total queries minus the used count.
        return m - used

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxRemoval([2,0,2], [[0,2],[0,2],[1,1]]))  # Expected output: 1
    print(sol.maxRemoval([1,1,1,1], [[1,3],[0,2],[1,3],[1,2]]))  # Expected output: 2
    print(sol.maxRemoval([1,2,3,4], [[0,3]]))  # Expected output: -1