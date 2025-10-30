class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        longest = 0
        for i in range(n):
            if i+1 < n and nums[i]+1 == nums[i+1]:
                length = 2
                j = i+2
                while j < n and (nums[j] == nums[i] if length % 2 == 0 else nums[j] == nums[j-1]+1):
                    length += 1
                    j += 1
                longest = max(longest, length)
        return longest if longest > 0 else -1