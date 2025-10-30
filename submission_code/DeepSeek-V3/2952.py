class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        # Calculate the initial sum of nums1
        total = sum(nums1)
        if total <= x:
            return 0
        # Pair nums1 and nums2 together and sort based on nums2
        pairs = list(zip(nums2, nums1))
        pairs.sort()
        # Initialize dp array
        # dp[t][k] represents the maximum reduction achievable in t seconds using the first k elements
        # We need to find the smallest t such that total - dp[t][n] <= x
        # Initialize dp with size (n+1) x (n+1)
        dp = [[0] * (n+1) for _ in range(n+1)]
        for t in range(1, n+1):
            for k in range(1, n+1):
                b, a = pairs[k-1]
                # Option 1: not take the k-th element
                option1 = dp[t][k-1]
                # Option 2: take the k-th element at time t
                option2 = dp[t-1][k-1] + a + b * t
                dp[t][k] = max(option1, option2)
        # Now, find the smallest t such that total - dp[t][n] <= x
        for t in range(n+1):
            if total - dp[t][n] <= x:
                return t
        return -1