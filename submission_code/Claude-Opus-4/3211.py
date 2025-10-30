class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        
        # dp[i] = maximum length of non-decreasing array using nums[0:i]
        dp = [0] * (n + 1)
        
        # last[i] = the value of the last element in the optimal array ending at position i
        last = [0] * (n + 1)
        
        # prefix sum for quick range sum calculation
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        # For optimization: maintain the best j for each position
        # We need to find j such that prefix[i] - prefix[j] >= last[j]
        # Rearranging: prefix[j] + last[j] <= prefix[i]
        
        # Use a monotonic deque to maintain candidates
        from collections import deque
        dq = deque([0])  # Store indices
        
        for i in range(1, n + 1):
            # Find the best j using binary search on deque
            while len(dq) > 1 and prefix[dq[1]] + last[dq[1]] <= prefix[i]:
                dq.popleft()
            
            j = dq[0]
            dp[i] = dp[j] + 1
            last[i] = prefix[i] - prefix[j]
            
            # Add current position to deque
            # Remove positions that will never be optimal
            while dq and prefix[i] + last[i] <= prefix[dq[-1]] + last[dq[-1]]:
                dq.pop()
            dq.append(i)
        
        return dp[n]