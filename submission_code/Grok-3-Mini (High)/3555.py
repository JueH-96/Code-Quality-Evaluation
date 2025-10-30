class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            min_index = min(range(len(nums)), key=lambda i: nums[i])
            nums[min_index] *= multiplier
        return nums