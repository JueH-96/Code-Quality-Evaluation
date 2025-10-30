class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        # Create pairs of (nums2[i], nums1[i]) sorted by nums2[i]
        pairs = sorted(zip(nums2, nums1))
        
        # dp[i][j] represents minimum sum possible after using j operations 
        # considering first i numbers
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        
        # For each prefix length
        for i in range(1, n + 1):
            inc, val = pairs[i-1]
            # For each number of operations
            for j in range(i + 1):
                # Don't reset current number
                dp[i][j] = dp[i-1][j] + inc * j + val
                
                # Reset current number
                if j > 0:
                    dp[i][j] = min(dp[i][j], dp[i-1][j-1])
        
        # Calculate initial sum
        total = sum(nums1)
        
        # For each time t
        for t in range(n + 1):
            # Calculate sum after t seconds
            curr_sum = total
            for i in range(n):
                curr_sum += nums2[i] * t
            
            # Check each possible combination of t operations
            for ops in range(t + 1):
                # Get minimum possible sum after ops operations
                min_sum = curr_sum - dp[n][ops]
                if min_sum <= x:
                    return t
                    
        return -1