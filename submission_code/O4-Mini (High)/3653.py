from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # min_prefix_sum[m] will track the minimum prefix sum seen so far
        # at indices i where i % k == m
        INF = 10**30
        min_prefix_sum = [INF] * k
        ans = -INF
        
        # running prefix sum
        prefix = 0
        
        # iterate through prefix sums at indices 0..n
        # prefix at index i is sum(nums[0..i-1])
        for i in range(n + 1):
            m = i % k
            if min_prefix_sum[m] == INF:
                # first time we see this mod class: initialize it
                min_prefix_sum[m] = prefix
            else:
                # we can form a subarray ending at i-1 whose length is divisible by k
                diff = prefix - min_prefix_sum[m]
                if diff > ans:
                    ans = diff
                # update the minimum prefix sum for this class
                if prefix < min_prefix_sum[m]:
                    min_prefix_sum[m] = prefix
            
            # add nums[i] to prefix for the next iteration
            if i < n:
                prefix += nums[i]
        
        return ans