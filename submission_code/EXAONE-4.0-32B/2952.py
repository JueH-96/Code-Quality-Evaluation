class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        totalA = sum(nums1)
        totalB = sum(nums2)
        
        B = sorted(nums2, reverse=True)
        C = 0
        for j in range(n):
            C += j * B[j]
        if C > x:
            return -1
        
        arr = [(nums1[i], nums2[i]) for i in range(n)]
        arr.sort(key=lambda x: x[1], reverse=True)
        
        for t in range(0, n + 1):
            base_val = totalA + t * totalB
            if base_val <= x:
                return t
                
            m = min(t, n)
            dp = [-10**18] * (m + 1)
            dp[0] = 0
            
            for i in range(n):
                a_i, b_i = arr[i]
                for j in range(m, 0, -1):
                    candidate = dp[j - 1] + a_i + b_i * (t - j + 1)
                    if candidate > dp[j]:
                        dp[j] = candidate
            
            for j in range(m + 1):
                if dp[j] >= base_val - x:
                    return t
                    
        return n