class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        s1 = sum(nums1)
        s2 = sum(nums2)
        n = len(nums1)
        if s1 <= x:
            return 0
        
        arr = list(zip(nums1, nums2))
        arr.sort(key=lambda x: x[1])
        
        dp = [0] * (n + 1)
        
        for i in range(n):
            a, b = arr[i]
            for j in range(min(i + 1, n), 0, -1):
                if dp[j] < dp[j - 1] + a + j * b:
                    dp[j] = dp[j - 1] + a + j * b
        
        for k in range(n + 1):
            total = s1 + k * s2 - dp[k]
            if total <= x:
                return k
        return -1