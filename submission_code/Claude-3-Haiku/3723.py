class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        total = 0
        for i in range(len(nums)):
            is_good = True
            if i - k >= 0 and nums[i] <= nums[i - k]:
                is_good = False
            if i + k < len(nums) and nums[i] <= nums[i + k]:
                is_good = False
            if is_good:
                total += nums[i]
        return total