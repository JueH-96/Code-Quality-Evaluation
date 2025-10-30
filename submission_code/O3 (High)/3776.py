from typing import List
from collections import defaultdict
import math

class Solution:
    def minCost(self, nums: List[int]) -> int:
        """
        Dynamic programming with at most two elements kept in a buffer.
        
        States:
            dp1  : minimal cost when the current array starts with the single element `x`
                   (buffer size == 1).   Key -> value of that element,  value -> minimal cost
                   
            dp2  : minimal cost when the current array starts with the two elements `(x, y)`
                   in this order (buffer size == 2). Key -> (x, y), value -> minimal cost
        
        Transition while reading the next element z:
            1) From dp1 (size-1 buffer) we append z  -> size-2 buffer (x, z)
               cost does not change.
               
            2) From dp2 (size-2 buffer) we append z  -> size-3, therefore one operation
               must be executed.  For every triple (x, y, z) we can keep any of the
               three elements.  If we keep k, the two removed elements cost
               max(other_two).  The kept element becomes the new single-element buffer.
        
        After the whole array is scanned the buffer is either of size-1 (dp1 not empty)
        or size-2 (dp2 not empty).  The last operation removes everything that is left,
        its cost equals the maximum element still present.
        """
        
        n = len(nums)
        if n == 1:
            return nums[0]                       # single final operation
        
        INF = math.inf
        
        # buffer of size 1 after reading the first element
        dp1 = {nums[0]: 0}
        dp2 = {}
        
        for idx in range(1, n):
            z = nums[idx]
            
            # the dictionaries that will store states after processing nums[idx]
            nxt1 = {}
            nxt2 = {}
            
            # 1) size-1  ->  size-2 (simply append z)
            for x, c in dp1.items():
                key = (x, z)
                if key not in nxt2 or c < nxt2[key]:
                    nxt2[key] = c
            
            # 2) size-2  ->  size-3  ->  perform one removal  ->  size-1
            for (x, y), c in dp2.items():
                # keep x, remove y & z
                cost = c + max(y, z)
                if x not in nxt1 or cost < nxt1[x]:
                    nxt1[x] = cost
                # keep y, remove x & z
                cost = c + max(x, z)
                if y not in nxt1 or cost < nxt1[y]:
                    nxt1[y] = cost
                # keep z, remove x & y
                cost = c + max(x, y)
                if z not in nxt1 or cost < nxt1[z]:
                    nxt1[z] = cost
            
            dp1, dp2 = nxt1, nxt2
        
        # Final removal (buffer has size 1 or 2)
        ans = INF
        for x, c in dp1.items():                 # buffer size 1
            ans = min(ans, c + x)
        for (x, y), c in dp2.items():            # buffer size 2
            ans = min(ans, c + max(x, y))
        
        return ans