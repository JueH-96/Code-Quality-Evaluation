from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        total_sum = sum(nums1)
        
        if sum(nums2) == 0:
            if total_sum <= x:
                return 0
            else:
                return -1
        
        indices = list(range(n))
        indices.sort(key=lambda i: nums2[i], reverse=True)
        
        for time in range(n + 1):
            current_sum = total_sum
            
            for i in range(n):
                current_sum += nums2[i] * time
            
            
            
            
            
            dp = [[float('inf')] * (time + 1) for _ in range(n + 1)]
            dp[0][0] = 0
            
            for i in range(1, n + 1):
                for j in range(time + 1):
                    dp[i][j] = dp[i-1][j]
                    if j > 0:
                        dp[i][j] = min(dp[i][j], dp[i-1][j-1] + nums1[indices[i-1]] + nums2[indices[i-1]] * time)
            
            
            min_sum = float('inf')
            for j in range(time + 1):
                min_sum = min(min_sum, dp[n][j])
            
            if current_sum - min_sum <= x:
                return time
        
        return -1