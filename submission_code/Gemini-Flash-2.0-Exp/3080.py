class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        i = 0
        while i < n:
            curr = nums[i]
            j = i
            while j < n and curr > 0:
                curr &= nums[j]
                j += 1
            if curr == 0:
                ans += 1
                i = j
            else:
                ans = 1
                break
        return ans