class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        import sys
        input_data = nums
        MOD = 10**9 + 7
        n = len(input_data)
        
        # Maximum possible value in nums, given constraints is 50
        max_val = max(input_data)
        
        # dp[i][a] = number of ways to fix arr1 up to index i with arr1[i] = a
        # subject to monotonic constraints for both arr1 and arr2.
        dp = [[0]*(max_val+1) for _ in range(n)]
        prefix = [[0]*(max_val+1) for _ in range(n)]
        
        # Base case for i = 0
        # arr1[0] can be anything from 0..nums[0]
        for a in range(input_data[0] + 1):
            dp[0][a] = 1
        
        # Build prefix sums for i=0
        prefix[0][0] = dp[0][0]
        for a in range(1, max_val+1):
            prefix[0][a] = (prefix[0][a-1] + dp[0][a]) % MOD
        
        # Fill dp for i=1..n-1
        for i in range(1, n):
            # D = max(0, nums[i] - nums[i-1])
            D = max(0, input_data[i] - input_data[i-1])
            
            for a in range(input_data[i] + 1):
                if a < D:
                    # It's impossible for arr1[i] to be a < D
                    dp[i][a] = 0
                else:
                    # We sum dp[i-1][b] for 0 <= b <= min(r, nums[i-1]),
                    # where r = a - D.
                    r = a - D
                    if r > input_data[i-1]:
                        r = input_data[i-1]
                    dp[i][a] = prefix[i-1][r]
                
                dp[i][a] %= MOD
            
            # Build prefix sums for dp[i]
            prefix[i][0] = dp[i][0]
            for a in range(1, max_val+1):
                prefix[i][a] = (prefix[i][a-1] + dp[i][a]) % MOD
        
        # The result is the sum of dp[n-1][a] for a in [0..nums[n-1]]
        return prefix[n-1][input_data[n-1]] % MOD