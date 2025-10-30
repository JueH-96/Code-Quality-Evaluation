class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        i = 0
        while i < n:
            j = i + 1
            while j < n and nums[j] != nums[j-1]:
                j += 1
            count += (j - i) * (j - i + 1) // 2
            i = j
        return count