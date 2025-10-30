from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        # mp will map a value v to the minimum prefix sum seen so far
        # at positions i where nums[i] == v
        mp = {}
        INF = 1 << 62
        
        n = len(nums)
        if n < 2:
            return 0
        
        # prefix is the running sum up to the previous index
        prefix = 0
        # initialize mapping with the first element at index 0
        # prefix before adding nums[0] is 0
        mp[nums[0]] = 0
        # now include nums[0] in the prefix
        prefix += nums[0]
        
        ans = -INF
        
        # iterate j from 1 to n-1
        for j in range(1, n):
            before = prefix
            after = before + nums[j]
            
            # we look for previous i with nums[i] == nums[j] + k or nums[j] - k
            t1 = nums[j] + k
            t2 = nums[j] - k
            if t1 in mp:
                ans = max(ans, after - mp[t1])
            if t2 in mp:
                ans = max(ans, after - mp[t2])
            
            # now update the map for nums[j] with the prefix sum before this index
            # so that future j' > j can use this
            if nums[j] in mp:
                mp[nums[j]] = min(mp[nums[j]], before)
            else:
                mp[nums[j]] = before
            
            prefix = after
        
        # if we never found a good subarray, return 0
        return ans if ans != -INF else 0