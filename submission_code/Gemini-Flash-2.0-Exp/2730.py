class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            original = nums[i]
            nums[i] = nums[i] * (1 << k)
            current_or = 0
            for j in range(n):
                current_or |= nums[j]
            ans = max(ans, current_or)
            nums[i] = original
        return ans