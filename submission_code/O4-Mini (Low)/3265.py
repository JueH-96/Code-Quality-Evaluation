from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        # prefix sum up to (but not including) current index
        pref = 0
        # map value -> minimum prefix sum at any index i where nums[i] == value
        min_pref = {}
        ans = float('-inf')
        found = False
        
        for x in nums:
            # consider ending a good subarray at the current element x
            # the start value must be x+k or x-k
            for target in (x + k, x - k):
                if target in min_pref:
                    found = True
                    # sum of subarray from some i to current index is (pref + x) - min_pref[target]
                    candidate = pref + x - min_pref[target]
                    ans = max(ans, candidate)
            
            # now update min_pref for starting a subarray at this index
            # the relevant prefix sum is the current pref (before adding x)
            if x not in min_pref:
                min_pref[x] = pref
            else:
                min_pref[x] = min(min_pref[x], pref)
            
            # include x in the running prefix sum
            pref += x
        
        # if no good subarray was found, return 0 as specified
        return ans if found else 0