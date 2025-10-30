from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Build prefix OR array
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] | nums[i]
        
        # Build suffix OR array
        suffix = [0] * (n + 1)
        for i in range(n-1, -1, -1):
            suffix[i] = suffix[i+1] | nums[i]
        
        answer = 0
        # Try applying all k shifts to each element
        for i in range(n):
            # OR of all elements except nums[i]
            without_i = prefix[i] | suffix[i+1]
            # Shift nums[i] by k (multiply by 2^k)
            candidate = without_i | (nums[i] << k)
            answer = max(answer, candidate)
        
        return answer