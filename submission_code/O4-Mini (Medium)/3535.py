from typing import List

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        # dp[i][v] = number of ways up to index i with arr1[i] = v
        # We only keep the previous row "prev" of dp
        # Initialize for i = 0
        prev = [1] * (nums[0] + 1)  # a[0] can be 0..nums[0], each way = 1
        
        for i in range(1, n):
            curr_max = nums[i]
            prev_max = nums[i - 1]
            # delta enforces arr2 non-increasing:
            # a[i] >= a[i-1] + max(0, nums[i] - nums[i-1])
            delta = max(0, nums[i] - nums[i - 1])
            
            # build prefix sums of prev for fast range-sum queries
            prefix = [0] * len(prev)
            running = 0
            for j in range(len(prev)):
                running = (running + prev[j]) % MOD
                prefix[j] = running
            
            # now compute current dp row
            curr = [0] * (curr_max + 1)
            for v in range(curr_max + 1):
                if v < delta:
                    # cannot satisfy the non-increasing constraint
                    curr[v] = 0
                else:
                    # sum over all u = 0..min(prev_max, v - delta)
                    u = v - delta
                    if u >= prev_max:
                        curr[v] = prefix[prev_max]
                    else:
                        curr[v] = prefix[u]
            prev = curr
        
        # answer is sum of dp at last index
        return sum(prev) % MOD