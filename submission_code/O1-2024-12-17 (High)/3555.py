class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            # Find the first occurrence of the minimum value
            min_val = min(nums)
            min_index = nums.index(min_val)
            # Replace it with min_val * multiplier
            nums[min_index] = min_val * multiplier
        return nums