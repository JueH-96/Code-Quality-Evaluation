class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        MAX_NUM = 51  # Since nums[i] can be up to 50
        n = len(nums)
        
        # Initialize dp and prefix arrays
        dp = [0] * MAX_NUM
        prefix = [0] * MAX_NUM
        
        for i in range(n):
            if i == 0:
                # Initialize dp for the first element
                for j in range(nums[0] + 1):
                    dp[j] = 1
                # Update prefix sums for the first row
                for j in range(MAX_NUM):
                    if j <= nums[0]:
                        prefix[j] = j + 1
                    else:
                        prefix[j] = prefix[nums[0]]
            else:
                new_dp = [0] * MAX_NUM
                # Compute new dp values based on previous prefix sums
                for j in range(nums[i] + 1):
                    delta = nums[i] - nums[i-1]
                    if delta > 0:
                        upper_k = j - delta
                        if upper_k >= 0:
                            new_dp[j] = prefix[upper_k]
                    else:
                        upper_k = j
                        new_dp[j] = prefix[upper_k]
                    new_dp[j] %= MOD
                # Update prefix sums for the next iteration
                temp_prefix = [0] * MAX_NUM
                running_sum = 0
                for j in range(MAX_NUM):
                    running_sum += new_dp[j]
                    running_sum %= MOD
                    temp_prefix[j] = running_sum
                dp = new_dp
                prefix = temp_prefix
        
        # Sum up the last dp row for the result
        answer = sum(dp) % MOD
        return answer