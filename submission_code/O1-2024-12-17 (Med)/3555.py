class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            # Find the minimum value in nums
            min_val = min(nums)
            # Find the index of the first occurrence of this minimum value
            idx = nums.index(min_val)
            # Multiply this value by the multiplier
            nums[idx] *= multiplier
        
        return nums