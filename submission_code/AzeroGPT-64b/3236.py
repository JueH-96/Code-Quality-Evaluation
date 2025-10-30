class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        sum_ = nums[0]
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:
                sum_ += nums[i]
            else:
                break

        for x in range(sum_, 100):
            if x not in nums: return x