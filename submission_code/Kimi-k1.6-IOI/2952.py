class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        sum_nums1 = sum(nums1)
        sum_nums2 = sum(nums2)
        
        dp = [-float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            a, b = nums1[i], nums2[i]
            # Update dp in reverse to avoid using the same element multiple times in the same step
            for k in range(n, 0, -1):
                if dp[k-1] != -float('inf'):
                    dp[k] = max(dp[k], dp[k-1] + a + k * b)
        
        min_time = -1
        for k in range(n + 1):
            if dp[k] != -float('inf'):
                total = sum_nums1 + sum_nums2 * k - dp[k]
                if total <= x:
                    if min_time == -1 or k < min_time:
                        min_time = k
        return min_time