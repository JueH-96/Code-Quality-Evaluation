class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = float('inf')
        for i in range(n):
            curr_or = 0
            for j in range(i, n):
                curr_or |= nums[j]
                if curr_or >= k:
                    ans = min(ans, j - i + 1)
                    break
        if ans == float('inf'):
            return -1
        else:
            return ans