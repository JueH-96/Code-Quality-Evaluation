class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        total = sum(nums1)
        pairs = sorted(zip(nums2, nums1), reverse=True)
        
        dp = [0] * (n + 1)
        
        for i, (rate, initial) in enumerate(pairs, 1):
            for j in range(i, 0, -1):
                dp[j] = max(dp[j], dp[j-1] + rate * j + initial)
        
        for time in range(n + 1):
            if total + sum(rate * time for rate, _ in pairs) - dp[time] <= x:
                return time
        
        return -1