class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        
        if m == 0:
            return 0
        
        # Precompute the overall AND of nums
        overall_and = nums[0]
        for num in nums[1:]:
            overall_and &= num
        
        # Compute the AND of all elements in andValues
        and_of_andValues = andValues[0]
        for val in andValues[1:]:
            and_of_andValues &= val
        
        # If the overall AND doesn't match, it's impossible
        if overall_and != and_of_andValues:
            return -1
        
        # Initialize the DP table
        INF = float('inf')
        dp = [ [INF] * (n + 1) for _ in range(m + 1) ]
        dp[0][0] = 0  # Base case: 0 subarrays, 0 elements processed
        
        for j in range(1, m + 1):
            target = andValues[j - 1]
            for s in range(n):
                current_and = None
                for e in range(s, n):
                    if e == s:
                        current_and = nums[e]
                    else:
                        current_and &= nums[e]
                    
                    if current_and == target:
                        # Update the DP table
                        if dp[j-1][s] != INF:
                            new_sum = dp[j-1][s] + nums[e]
                            if new_sum < dp[j][e + 1]:
                                dp[j][e + 1] = new_sum
                    elif current_and < target:
                        break  # No need to proceed further for this s
        
        if dp[m][n] == INF:
            return -1
        else:
            return dp[m][n]