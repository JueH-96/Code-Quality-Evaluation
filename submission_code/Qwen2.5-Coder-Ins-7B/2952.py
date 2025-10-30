class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        total = sum(nums1)
        pairs = sorted(zip(nums2, nums1))
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] + pairs[i - 1][1] + (i - 1) * pairs[i - 1][0])
        
        for t in range(n + 1):
            if dp[t][n] + total - dp[t][n] * t <= x:
                return t
        
        return -1