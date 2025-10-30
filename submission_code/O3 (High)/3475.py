from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        m = n - 2                    # number of possible starting positions of a length–3 flip
        
        # b[i] == 1  ⇔  position i has to be flipped an odd number of times
        b = [v ^ 1 for v in nums]    # we need every entry to become 1
        
        x = [0] * m                  # x[i] == 1  ⇔  we flip the triple starting at i
        
        # 1.  Determine x[0] … x[m-1] from the first n-2 equations
        if m >= 1:
            x[0] = b[0]
        if m >= 2:
            x[1] = b[1] ^ x[0]
            for i in range(2, m):
                x[i] = b[i] ^ x[i - 1] ^ x[i - 2]
        
        # 2.  Check the two remaining (over-determined) equations
        #     j = n-2  ⇒  x[m-1] ⊕ x[m-2] must equal b[n-2]
        lhs = x[m - 1] ^ (x[m - 2] if n >= 4 else 0)
        if lhs != b[n - 2]:
            return -1
        
        #     j = n-1  ⇒  x[m-1] must equal b[n-1]
        if x[m - 1] != b[n - 1]:
            return -1
        
        # 3.  The unique feasible x (if it exists) already has the minimum number of 1-bits
        return sum(x)