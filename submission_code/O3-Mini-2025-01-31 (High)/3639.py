from typing import List

class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        # Create a difference array for range updates (queries)
        diff = [0] * (n + 1)
        
        # For each query, add +1 at l and -1 at r+1 so that the prefix sum gives
        # the number of times an index is covered by a query.
        for l, r in queries:
            diff[l] += 1
            diff[r+1] -= 1
        
        # Now, compute the prefix sum to get the number of decrements available for each index.
        current = 0
        for i in range(n):
            current += diff[i]
            # Each nums[i] requires exactly nums[i] decrements.
            # Since each query can contribute at most one decrement per index,
            # we must have at least nums[i] available decrements at index i.
            if current < nums[i]:
                return False
        return True