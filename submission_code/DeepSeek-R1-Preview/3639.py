from typing import List

class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        q = len(queries)
        
        # Compute for each index the number of queries covering it
        d = [0] * n
        for l, r in queries:
            for i in range(l, r + 1):
                d[i] += 1
        
        # Check if any index has d[i] < nums[i]
        for i in range(n):
            if d[i] < nums[i]:
                return False
        
        # Process queries in reverse order
        c = nums.copy()
        for l, r in reversed(queries):
            for i in range(l, r + 1):
                if c[i] > 0:
                    c[i] -= 1
        
        # Check if all elements are zero
        return all(x == 0 for x in c)