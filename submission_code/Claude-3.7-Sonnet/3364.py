class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)
        
        # memo[i][j] represents the minimum possible sum for dividing nums[0:i] into j subarrays
        memo = {}
        
        def dp(i, j):
            # Base cases
            if j == 0:
                return 0 if i == 0 else float('inf')
            
            if i < j:  # Can't divide i elements into j > i subarrays
                return float('inf')
            
            if (i, j) in memo:
                return memo[i, j]
            
            min_sum = float('inf')
            
            # Try all possible starting positions for the jth subarray
            for k in range(j - 1, i):
                # Calculate the bitwise AND of nums[k:i]
                current_and = nums[k]
                for idx in range(k + 1, i):
                    current_and &= nums[idx]
                
                # Check if this subarray's AND matches what we need
                if current_and == andValues[j - 1]:
                    prev_sum = dp(k, j - 1)
                    if prev_sum != float('inf'):
                        # The value of this subarray is its last element (nums[i-1])
                        min_sum = min(min_sum, prev_sum + nums[i - 1])
            
            memo[i, j] = min_sum
            return min_sum
        
        result = dp(n, m)
        return result if result != float('inf') else -1