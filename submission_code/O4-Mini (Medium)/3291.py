from typing import List

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        # Helper to count set bits
        def bitcount(x: int) -> int:
            return bin(x).count('1')
        
        n = len(nums)
        # Precompute bit counts
        b = [bitcount(x) for x in nums]
        # Fully sorted array
        sorted_nums = sorted(nums)
        
        # Find contiguous segments of equal bitcount in the original array
        i = 0
        while i < n:
            j = i
            while j + 1 < n and b[j + 1] == b[i]:
                j += 1
            # Now [i..j] is one segment
            orig_seg = nums[i:j+1]
            target_seg = sorted_nums[i:j+1]
            # We can permute freely inside this segment,
            # so the multisets must match
            if sorted(orig_seg) != sorted(target_seg):
                return False
            i = j + 1
        
        return True