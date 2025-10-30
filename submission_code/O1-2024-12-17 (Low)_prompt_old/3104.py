class Solution:
    def countWays(self, nums: List[int]) -> int:
        import sys
        from typing import List
        
        n = len(nums)
        
        # Frequency array: freq[x] = number of students who have nums[i] = x
        # We use size n+1 so we can safely refer to freq[n] (which will stay 0).
        freq = [0]*(n+1)
        for x in nums:
            freq[x] += 1
        
        # Prefix sums: prefix[k] = freq[0] + freq[1] + ... + freq[k-1]
        prefix = [0]*(n+2)
        for i in range(n+1):
            prefix[i+1] = prefix[i] + freq[i]
        
        # We count how many k in [0..n] satisfy:
        #   freq[k] = 0  AND  prefix[k] = k
        # Each valid k gives exactly 1 way (choose exactly those with nums[i]<k).
        ans = 0
        for k in range(n+1):
            # freq[k] might be out of range if k==n, but we've extended freq to n+1,
            # so freq[n] is valid and is 0 by construction (since nums[i] < n).
            if freq[k] == 0 and prefix[k] == k:
                ans += 1
        
        return ans