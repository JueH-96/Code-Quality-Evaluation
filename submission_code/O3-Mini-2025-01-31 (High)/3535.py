from typing import List

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        
        # dp[x] will denote the number of valid ways to have a[i] = x 
        # for the current index, where x is between 0 and nums[i].
        # Since nums[i] is always <= 50, we use an array of size 51.
        dp = [0] * 51
        
        # For the first element, a[0] can be any value from 0 to nums[0].
        for x in range(nums[0] + 1):
            dp[x] = 1
        
        # Process indices from 0 to n-2, computing dp for the next index.
        for i in range(n - 1):
            current_max = nums[i]      # allowed a[i] values: 0..nums[i]
            next_max = nums[i+1]         # allowed a[i+1] values: 0..nums[i+1]
            # The condition on b's (monotone non-increasing) forces:
            # a[i+1] - a[i] >= nums[i+1] - nums[i] if nums[i+1] >= nums[i],
            # otherwise no extra lower bound apart from a[i+1] >= a[i].
            # So let diff = max(0, nums[i+1] - nums[i]).
            diff = max(0, nums[i+1] - nums[i])
            
            # Build a prefix sum array for dp[x] over allowed x values for index i.
            prefix = [0] * (current_max + 1)
            prefix[0] = dp[0] % mod
            for x in range(1, current_max + 1):
                prefix[x] = (prefix[x-1] + dp[x]) % mod
            
            # Compute the new dp state for index i+1.
            new_dp = [0] * 51
            # a[i+1] can be any value y from 0 to next_max.
            # But to satisfy the transition condition, if we came from some a[i] = x, we need:
            #   y >= x + diff  or  x <= y - diff.
            # So for a fixed y, we consider all x in [0, min(current_max, y-diff)].
            for y in range(next_max + 1):
                if y < diff:
                    new_dp[y] = 0
                else:
                    x_max = y - diff  # maximum x such that x <= y - diff.
                    if x_max > current_max:
                        x_max = current_max
                    new_dp[y] = prefix[x_max]  # sum of dp[x] for x in [0, x_max].
            dp = new_dp  # update dp to be our new state
        
        # The final answer is the sum of the ways for the last index over the allowed a[n-1] values.
        return sum(dp[:nums[-1] + 1]) % mod