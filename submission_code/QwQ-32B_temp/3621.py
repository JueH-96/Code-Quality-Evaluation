from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Check if any element is less than k and not equal to k, which makes it impossible
        for num in nums:
            if num < k and num != k:
                return -1
        
        # Collect elements greater than k and count distinct values
        greater = [num for num in nums if num > k]
        if not greater:
            return 0
        return len(set(greater))