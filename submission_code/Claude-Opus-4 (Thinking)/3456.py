class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # dp[j][v] = max length of subsequence with j transitions ending with value v
        dp = [defaultdict(int) for _ in range(k + 1)]
        
        for num in nums:
            new_dp = [defaultdict(int) for _ in range(k + 1)]
            
            # Start a new subsequence
            new_dp[0][num] = 1
            
            # Extend existing subsequences
            for j in range(k + 1):
                for v, length in dp[j].items():
                    # Don't include current element
                    new_dp[j][v] = max(new_dp[j][v], length)
                    
                    # Include current element
                    if v == num:
                        # No new transition
                        new_dp[j][num] = max(new_dp[j][num], length + 1)
                    elif j < k:
                        # New transition
                        new_dp[j + 1][num] = max(new_dp[j + 1][num], length + 1)
            
            dp = new_dp
        
        # Find the maximum length across all states
        max_length = 0
        for j in range(k + 1):
            if dp[j]:
                max_length = max(max_length, max(dp[j].values()))
        
        return max_length