class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        import sys
        input_data = nums
        
        mod = 10**9 + 7
        n = len(input_data)
        
        # dp[i][v] = number of ways to construct arr1 up to index i with arr1[i] = v
        # subject to arr1[j] + arr2[j] = nums[j], arr1 non-decreasing, arr2 non-increasing
        # We'll only need dp for values v in [0..50] since nums[i] <= 50.
        
        dp = [[0]*51 for _ in range(n)]
        
        # Initialize dp[0][v] = 1 for v in [0..nums[0]]
        for v in range(input_data[0] + 1):
            dp[0][v] = 1
        
        # Fill dp using the transition:
        # dp[i+1][y] = sum_{all x} dp[i][x] where x satisfies the monotonic constraints:
        #   arr1[i+1] >= arr1[i]   => y >= x
        #   arr2[i] >= arr2[i+1]  => (nums[i] - x) >= (nums[i+1] - y)
        # => y - x >= nums[i+1] - nums[i]
        # => y >= x + (nums[i+1] - nums[i])  (or equivalently x <= y - (nums[i+1]-nums[i]) )
        #
        # Offset = max(0, nums[i+1] - nums[i])
        # Then x in [0..nums[i]], x <= y - offset. We'll sum dp[i][x] for all feasible x.
        
        for i in range(n - 1):
            # Compute prefix sums for dp[i] to speed up transitions
            prefix = [0]*52
            for v in range(51):
                prefix[v+1] = (prefix[v] + dp[i][v]) % mod
            
            offset = max(0, input_data[i+1] - input_data[i])
            
            for y in range(input_data[i+1] + 1):
                # We need x <= y - offset and x >= 0 and x <= nums[i]
                high = y - offset
                if high < 0:
                    # No valid x
                    dp[i+1][y] = 0
                else:
                    if high > input_data[i]:
                        high = input_data[i]
                    # sum of dp[i][0..high]
                    dp[i+1][y] = prefix[high+1] % mod
        
        # The answer is sum of dp[n-1][v] for v in [0..nums[n-1]]
        return sum(dp[n-1][v] for v in range(input_data[n-1] + 1)) % mod