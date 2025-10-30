class Solution:
    def minElement(self, nums: List[int]) -> int:
        for i in range(len(nums)):
            num = nums[i]
            digit_sum = 0
            while num > 0:
                digit_sum += num % 10
                num //= 10
            nums[i] = digit_sum
        return min(nums)