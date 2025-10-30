from collections import defaultdict

MOD = 10**9 + 7

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        # Initialize a dictionary to store the sum and count of subsequences ending with each number
        dp = defaultdict(lambda: [0, 0])
        
        for num in nums:
            # Initialize the current sum and count for the current number
            current_sum = num
            current_count = 1
            
            # Check for the previous number that is either num-1 or num+1
            for prev_num in [num - 1, num + 1]:
                if prev_num in dp:
                    # Add the sum and count of subsequences ending with prev_num
                    current_sum += dp[prev_num][0] + dp[prev_num][1] * num
                    current_count += dp[prev_num][1]
            
            # Update the dp for the current number
            dp[num][0] = (dp[num][0] + current_sum) % MOD
            dp[num][1] = (dp[num][1] + current_count) % MOD
        
        # Calculate the total sum by summing up all the sums in dp
        total_sum = 0
        for key in dp:
            total_sum = (total_sum + dp[key][0]) % MOD
        
        return total_sum