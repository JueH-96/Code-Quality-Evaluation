class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        count = 0
        for i in range(len(nums)):
            if i % 2 == 0:
                if nums[i] == nums[i-1]:
                    count += 1
            else:
                if nums[i] != nums[i-1]:
                    count += 1
        return count