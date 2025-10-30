class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # dp[i] will store the number of good partitions for nums[0:i+1]
        dp = [0] * (n + 1)
        dp[0] = 1  # Base case: one way to partition an empty array
        
        # last_occurrence will store the last occurrence index of each number
        last_occurrence = {}
        
        for i in range(1, n + 1):
            num = nums[i - 1]
            
            # Start with the number of partitions up to the previous element
            dp[i] = dp[i - 1]
            
            # If the current number has appeared before, we need to subtract
            # the partitions that would cause a duplicate in the current subarray
            if num in last_occurrence:
                last_index = last_occurrence[num]
                dp[i] = (dp[i] - dp[last_index - 1]) % MOD
            
            # Update the last occurrence of the current number
            last_occurrence[num] = i
        
        # The result is the number of partitions for the entire array
        return dp[n]