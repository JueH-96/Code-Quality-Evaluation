class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        l, r = 0, len(nums) - 1
        total_sum = 0
        ans = 0
        while r >= l:
            total_sum += nums[r]
            while (r - l + 1) * nums[r] - total_sum > k * numOperations:
                total_sum -= nums[l]
                l += 1
            ans = max(ans, r - l + 1)
            r -= 1
        return ans