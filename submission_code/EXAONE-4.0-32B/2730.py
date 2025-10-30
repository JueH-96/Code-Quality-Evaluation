class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        suf = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suf[i] = suf[i + 1] | nums[i]
        
        pre = 0
        ans = 0
        for i in range(n):
            candidate = (nums[i] << k) | pre | suf[i + 1]
            if candidate > ans:
                ans = candidate
            pre |= nums[i]
        
        return ans