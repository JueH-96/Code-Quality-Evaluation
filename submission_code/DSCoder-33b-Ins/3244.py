class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        nums.sort()
        count = 1
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                count += 1
        return count