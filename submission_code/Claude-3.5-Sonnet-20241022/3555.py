class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        nums = nums.copy()
        for _ in range(k):
            # Find index of first minimum value
            min_idx = 0
            for i in range(1, len(nums)):
                if nums[i] < nums[min_idx]:
                    min_idx = i
            
            # Multiply minimum value by multiplier
            nums[min_idx] *= multiplier
            
        return nums