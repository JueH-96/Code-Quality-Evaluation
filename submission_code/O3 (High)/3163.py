from typing import List

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        """
        For every start index i we expand the end index j,
        maintain the current set of distinct elements,
        and accumulate (len(distinct))² for the subarray nums[i..j].
        With n ≤ 100 this O(n²) approach is easily fast enough.
        """
        n = len(nums)
        total = 0
        
        # Iterate over all starting indices.
        for i in range(n):
            distinct = set()
            # Extend the subarray to every possible end index.
            for j in range(i, n):
                distinct.add(nums[j])
                d = len(distinct)
                total += d * d
                
        return total