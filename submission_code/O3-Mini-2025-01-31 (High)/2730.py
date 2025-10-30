from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Precompute prefix OR: prefix[i] is the OR from nums[0] to nums[i]
        prefix = [0] * n
        prefix[0] = nums[0]
        for i in range(1, n):
            prefix[i] = prefix[i - 1] | nums[i]
        
        # Precompute suffix OR: suffix[i] is the OR from nums[i] to nums[n - 1]
        suffix = [0] * n
        suffix[n - 1] = nums[n - 1]
        for i in range(n - 2, -1, -1):
            suffix[i] = suffix[i + 1] | nums[i]
        
        # factor to multiply by is equivalent to left shifting by k
        factor = 1 << k  # 2^k
        
        res = 0
        
        # Try applying all operations (i.e., shifting the chosen element by k)
        # to every index, and take the OR with all the other elements as they are.
        for i in range(n):
            candidate = 0
            # OR all elements to the left
            if i > 0:
                candidate |= prefix[i - 1]
            # Apply k operations on the current element (multiply by 2^k)
            candidate |= (nums[i] * factor)
            # OR all elements to the right
            if i < n - 1:
                candidate |= suffix[i + 1]
            res = max(res, candidate)
        
        return res