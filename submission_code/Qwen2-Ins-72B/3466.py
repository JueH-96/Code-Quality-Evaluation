class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            mask = 0
            for j in range(i, n):
                mask &= nums[j]
                if mask == k:
                    ans += 1
        return ans