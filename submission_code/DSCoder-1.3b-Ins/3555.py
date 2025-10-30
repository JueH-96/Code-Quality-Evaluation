class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        while k > 0:
            min_val = min(nums)
            min_index = nums.index(min_val)
            nums[min_index] = min_val * multiplier
            k -= 1
        return nums