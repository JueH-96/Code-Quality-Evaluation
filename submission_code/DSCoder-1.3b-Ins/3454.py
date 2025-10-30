class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        nums.sort()
        target.sort()
        ans = 0
        j = 0
        for i in range(n):
            while j < n and nums[j] <= target[i]:
                j += 1
            ans += j - i - 1
        return ans