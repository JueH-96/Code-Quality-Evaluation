class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            j = i
            while j + 1 < n and nums[j+1] != nums[j]:
                j += 1
            count += (j - i + 1)
        return count