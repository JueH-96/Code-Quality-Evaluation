class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n, m = len(nums), len(pattern)
        ans = 0
        for i in range(n - m):
            if all(pattern[j] in (0, nums[i + j + 1] - nums[i + j]) for j in range(m)):
                ans += 1
        return ans