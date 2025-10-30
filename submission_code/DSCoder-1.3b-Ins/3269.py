class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        count = 0
        for i in range(len(nums)):
            if nums[i:i+len(pattern)+1] == pattern:
                count += 1
        return count