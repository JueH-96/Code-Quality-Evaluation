class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            # Find index of first occurrence of the minimum value
            min_value = min(nums)
            min_index = nums.index(min_value)
            # Multiply that minimum value by 'multiplier'
            nums[min_index] *= multiplier
        return nums