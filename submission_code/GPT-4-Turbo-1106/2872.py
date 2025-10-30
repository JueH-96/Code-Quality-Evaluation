class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        while True:
            found = False
            for i in range(len(nums) - 1):
                if nums[i] <= nums[i + 1]:
                    nums[i + 1] += nums[i]
                    nums.pop(i)
                    found = True
                    break
            if not found:
                break
        return max(nums)