class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        ans = val = 0
        for v in nums:
            val &= v
            if val == 0:
                ans += 1
                val = (1 << 25) - 1
        return ans