class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        
        # Helper function to calculate the maximum sum of subarrays of at least length m
        def maxSubarraySumAtLeastM(start, end, m):
            if end - start + 1 < m:
                return float('-inf')
            
            # Calculate prefix sums
            prefix_sums = [0] * (end - start + 2)
            for i in range(start, end + 1):
                prefix_sums[i - start + 1] = prefix_sums[i - start] + nums[i]
            
            # Use a deque to find the max sum of subarrays of at least length m
            from collections import deque
            max_sum = float('-inf')
            deque_min = deque()  # stores indices where we can start taking subarray sums
            for i in range(m, end - start + 2):
                # We want to maintain the deque such that it contains indices of prefix_sums
                # which can form subarrays of at least length m
                while deque_min and prefix_sums[i - m] < prefix_sums[deque_min[-1]]:
                    deque_min.pop()
                deque_min.append(i - m)
                
                # Calculate the current subarray sum and update max_sum
                current_sum = prefix_sums[i] - prefix_sums[deque_min[0]]
                max_sum = max(max_sum, current_sum)
                
                # Remove elements from deque_min that are out of the current window
                if deque_min[0] == i - m:
                    deque_min.popleft()
            
            return max_sum
        
        # DP table where dp[i][j] represents the maximum sum of j subarrays from first i elements
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        
        # Initialize dp table
        for j in range(1, k + 1):
            for i in range(1, n + 1):
                dp[i][j] = float('-inf')
                if i >= m:
                    for l in range(m, i + 1):  # l is the length of the last subarray
                        dp[i][j] = max(dp[i][j], dp[i - l][j - 1] + maxSubarraySumAtLeastM(i - l, i - 1, m))
        
        return dp[n][k]