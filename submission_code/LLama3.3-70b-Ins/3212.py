from typing import List

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Initialize a dictionary to store the last seen index of each number
        last_seen = {}
        
        # Initialize a list to store the number of good partitions ending at each index
        dp = [0] * n
        
        # Initialize the number of good partitions ending at index 0
        dp[0] = 1
        
        # Iterate over the array
        for i in range(1, n):
            # Initialize the number of good partitions ending at index i
            dp[i] = dp[i-1]
            
            # Check if the current number has been seen before
            if nums[i] in last_seen:
                # If the current number has been seen before, update the number of good partitions
                # ending at index i by subtracting the number of good partitions ending at the last seen index
                dp[i] -= dp[last_seen[nums[i]] - 1]
            
            # Update the last seen index of the current number
            last_seen[nums[i]] = i + 1
            
            # Update the number of good partitions ending at index i by adding the number of good partitions
            # ending at the previous index
            dp[i] += dp[i-1]
            
            # Take the modulus to avoid overflow
            dp[i] %= MOD
        
        # Return the number of good partitions ending at the last index
        return dp[-1]