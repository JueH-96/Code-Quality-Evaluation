class Solution:
    def isGood(self, nums: List[int]) -> bool:
        from collections import Counter
        
        n = max(nums)
        count = Counter(nums)
        
        # Check if the maximum number appears exactly twice
        if count[n] != 2:
            return False
        
        # Check if all numbers from 1 to n-1 appear exactly once
        for i in range(1, n):
            if count[i] != 1:
                return False
        
        # Check if the length matches
        if len(nums) != n + 1:
            return False
        
        return True