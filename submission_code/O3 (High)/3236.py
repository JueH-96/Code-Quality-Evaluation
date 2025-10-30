from typing import List

class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # 1) Find longest sequential prefix starting from index 0
        seq_sum = nums[0]                   # sum of the prefix
        last_val = nums[0]                  # last value in the current prefix
        
        for i in range(1, len(nums)):
            if nums[i] == last_val + 1:     # still sequential
                seq_sum += nums[i]
                last_val = nums[i]
            else:                           # prefix breaks
                break
        
        # 2) Identify the smallest integer ≥ seq_sum that is NOT present in nums
        present = set(nums)
        candidate = seq_sum
        while candidate in present:
            candidate += 1
        
        return candidate