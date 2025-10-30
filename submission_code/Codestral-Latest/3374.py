class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        i = 0

        while i < n:
            j = i
            while j < n - 1 and nums[j] != nums[j + 1]:
                j += 1
            count += (j - i + 1) * (j - i + 2) // 2
            i = j + 1

        return count