from typing import List

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        
        # dp_prev[v] = number of ways for arr1[0..i-1] ending with arr1[i-1]=v
        dp_prev = [1] * (nums[0] + 1)
        
        for i in range(1, n):
            # Minimum increment needed at step i-1 -> i to keep arr2 non-increasing
            t = max(0, nums[i] - nums[i-1])
            max_prev = nums[i-1]
            
            # Build prefix sums of dp_prev for fast range-sum queries
            prefix = [0] * (max_prev + 1)
            running = 0
            for u in range(max_prev + 1):
                running = (running + dp_prev[u]) % mod
                prefix[u] = running
            
            # Compute dp_curr for position i
            m = nums[i]
            dp_curr = [0] * (m + 1)
            for v in range(m + 1):
                lim = v - t
                if lim < 0:
                    continue
                if lim > max_prev:
                    lim = max_prev
                dp_curr[v] = prefix[lim]
            
            dp_prev = dp_curr
        
        # Sum over all possible ending values at position n-1
        return sum(dp_prev) % mod