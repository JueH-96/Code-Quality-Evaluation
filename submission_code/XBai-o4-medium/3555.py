class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            min_val = min(nums)
            index = nums.index(min_val)
            nums[index] *= multiplier
        return nums