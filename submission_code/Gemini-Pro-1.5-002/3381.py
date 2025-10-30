class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = float('inf')

        for i in range(n):
            curr = 0
            for j in range(i, n):
                curr |= nums[j]
                if curr >= k:
                    ans = min(ans, j - i + 1)
                    break
        
        return ans if ans != float('inf') else -1