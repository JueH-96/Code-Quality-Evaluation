class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        ans = -float('inf')
        for i in range(1, 1 << n):
            prod = 1
            for j in range(n):
                if (i >> j) & 1:
                    prod *= nums[j]
            ans = max(ans, prod)
        return ans