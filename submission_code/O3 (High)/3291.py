from typing import List

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        """
        An element is allowed to swap only with an adjacent element that has the same
        number of set bits. Hence two elements that have different bit–counts can
        never cross each other – their relative order is fixed for the whole
        process.  Consequently, the array can only be rearranged **inside**
        every contiguous block whose elements all share the same bit–count.
        
        Algorithm:
        1.  Split the array into these maximal contiguous blocks.
        2.  Individually sort every block (any order inside a block is reachable,
            because all its adjacent pairs are swappable).
        3.  Concatenate the sorted blocks.  
            If that final array is globally sorted, the original array can be
            sorted; otherwise it cannot.
        """
        n = len(nums)
        # Target sorted order
        fully_sorted = sorted(nums)

        # Pre–compute pop-counts once
        bits = [bin(x).count("1") for x in nums]

        i = 0
        rebuilt = []
        while i < n:
            j = i
            # gather one maximal contiguous block with identical pop-count
            while j < n and bits[j] == bits[i]:
                j += 1
            block = nums[i:j]
            block.sort()
            rebuilt.extend(block)
            i = j

        return rebuilt == fully_sorted