class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # dp[i][j] = set of possible OR values when selecting j elements from first i elements
        dp = [[set() for _ in range(k + 1)] for _ in range(n + 1)]
        
        # Base case: selecting 0 elements gives OR value 0
        for i in range(n + 1):
            dp[i][0].add(0)
        
        # Fill the DP table
        for i in range(1, n + 1):
            for j in range(k + 1):
                # Don't take nums[i-1]
                dp[i][j] = dp[i][j] | dp[i-1][j]
                
                # Take nums[i-1]
                if j > 0:
                    for or_val in dp[i-1][j-1]:
                        dp[i][j].add(or_val | nums[i-1])
        
        max_val = 0
        
        # Try all possible split points
        # We need at least k elements before position i and at least k elements from position i
        for i in range(k, n - k + 1):
            # Get all possible OR values for first k elements from first i elements
            left_ors = dp[i][k]
            
            # Calculate possible OR values for k elements from position i onwards
            # dp2[j][cnt] = set of possible OR values selecting cnt elements from nums[i:i+j]
            dp2 = [[set() for _ in range(k + 1)] for _ in range(n - i + 1)]
            
            for j in range(n - i + 1):
                dp2[j][0].add(0)
            
            for j in range(1, n - i + 1):
                for cnt in range(k + 1):
                    # Don't take nums[i+j-1]
                    dp2[j][cnt] = dp2[j][cnt] | dp2[j-1][cnt]
                    
                    # Take nums[i+j-1]
                    if cnt > 0:
                        for or_val in dp2[j-1][cnt-1]:
                            dp2[j][cnt].add(or_val | nums[i+j-1])
            
            right_ors = dp2[n-i][k]
            
            # Calculate maximum XOR value
            for left_or in left_ors:
                for right_or in right_ors:
                    max_val = max(max_val, left_or ^ right_or)
        
        return max_val