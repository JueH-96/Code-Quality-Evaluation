class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        
        # Initialize the possible ranges for arr1 and arr2
        # For each index i, arr1[i] can be from 0 to nums[i]
        # arr2[i] will be nums[i] - arr1[i]
        
        # We need to ensure that arr1 is non-decreasing and arr2 is non-increasing
        
        # To find the number of valid (arr1, arr2) pairs, we can model this as a dynamic programming problem
        # where we track the possible values of arr1[i] and ensure the constraints are met
        
        # Let dp[i][x] represent the number of ways to assign arr1[0..i] such that arr1[i] = x and the constraints are satisfied
        
        # Initialize dp[0][x] for all x from 0 to nums[0]
        dp = [dict() for _ in range(n)]
        for x in range(nums[0] + 1):
            dp[0][x] = 1
        
        for i in range(1, n):
            new_dp = dict()
            for prev_x in dp[i-1]:
                for x in range(prev_x, nums[i] + 1):
                    # Ensure that arr2[i] = nums[i] - x is >= arr2[i+1] if i+1 < n
                    # But since we are processing in order, we need to ensure that arr2[i] is <= arr2[i-1]
                    # arr2[i] = nums[i] - x
                    # arr2[i-1] = nums[i-1] - prev_x
                    # So nums[i] - x <= nums[i-1] - prev_x
                    if nums[i] - x <= nums[i-1] - prev_x:
                        if x in new_dp:
                            new_dp[x] = (new_dp[x] + dp[i-1][prev_x]) % MOD
                        else:
                            new_dp[x] = dp[i-1][prev_x] % MOD
            dp[i] = new_dp
        
        # Sum all possible ways for the last element
        total = 0
        for x in dp[-1]:
            total = (total + dp[-1][x]) % MOD
        return total