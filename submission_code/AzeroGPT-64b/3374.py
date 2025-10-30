class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        count = 1
        for i in range(1, n):
            if nums[i] != nums[i-1]:
                count += 1
            ans += count
        return ans