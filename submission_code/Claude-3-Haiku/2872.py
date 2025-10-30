class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        while len(nums) > 1:
            max_idx = 0
            for i in range(1, len(nums)):
                if nums[i] > nums[max_idx]:
                    max_idx = i
            if max_idx > 0 and nums[max_idx-1] <= nums[max_idx]:
                nums[max_idx-1] += nums[max_idx]
                nums.pop(max_idx)
            else:
                break
        return nums[0]