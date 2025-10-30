from typing import List

class BIT:
    def __init__(self, n: int):
        self.n = n
        self.tree = [0]*(n+1)
    
    def update(self, index: int, delta: int) -> None:
        # BIT index is 1-indexed
        i = index + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def query(self, index: int) -> int:
        # sum from 0 to index (inclusive)
        res = 0
        i = index + 1
        while i:
            res += self.tree[i]
            i -= i & -i
        return res

    def range_query(self, left: int, right: int) -> int:
        if left > right:
            return 0
        return self.query(right) - (self.query(left-1) if left > 0 else 0)

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        # Helper function to determine if index i is a peak.
        # Only valid if 0 < i < n-1
        def is_peak(i: int) -> int:
            if i <= 0 or i >= n-1:
                return 0
            return 1 if nums[i] > nums[i-1] and nums[i] > nums[i+1] else 0
        
        # Build initial peaks array
        peaks = [0]*n
        for i in range(1, n-1):
            peaks[i] = 1 if (nums[i] > nums[i-1] and nums[i] > nums[i+1]) else 0
        
        bit = BIT(n)
        # Build BIT: Update for each valid index i
        for i in range(n):
            if peaks[i]:
                bit.update(i, peaks[i])
        
        result = []
        # Process each query:
        for query in queries:
            if query[0] == 1:
                # Query type1: count peaks in subarray
                l, r = query[1], query[2]
                # The first and last element in the subarray cannot be peaks.
                if r - l < 2:
                    result.append(0)
                else:
                    # Effective range is [l+1, r-1]
                    count = bit.range_query(l+1, r-1)
                    result.append(count)
            else:
                # Query type 2: update a value
                index, val = query[1], query[2]
                # Update the nums array
                nums[index] = val
                
                # Indices that might change: index-1, index, index+1
                for i in [index - 1, index, index + 1]:
                    if 1 <= i < n-1:  # only valid indices for peaks
                        new_val = 1 if (nums[i] > nums[i-1] and nums[i] > nums[i+1]) else 0
                        if new_val != peaks[i]:
                            # Update BIT with the difference and update peaks array value.
                            delta = new_val - peaks[i]
                            bit.update(i, delta)
                            peaks[i] = new_val
        return result