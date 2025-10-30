class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        length = 1
        ans = 1
        n = len(nums)
        for i in range(1, n):
            if nums[i] != nums[i-1]:
                length += 1
            else:
                length = 1
            ans += length
        return ans