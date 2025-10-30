class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            min_index = nums.index(min(nums))  # Find the index of the first minimum value
            nums[min_index] *= multiplier  # Replace it with its multiplied value
        return nums