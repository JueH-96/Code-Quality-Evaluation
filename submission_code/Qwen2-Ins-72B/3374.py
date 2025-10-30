class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        total = 0
        i = 0
        while i < len(nums):
            j = i + 1
            while j < len(nums) and nums[j-1] != nums[j]:
                j += 1
            length = j - i
            total += (1 + length) * length // 2
            i = j
        return total