from collections import Counter

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        if not nums:
            return False  # Though constraints say len >=1, handle empty case for robustness
        
        max_val = max(nums)
        n = max_val
        
        if len(nums) != n + 1:
            return False
        
        counts = Counter(nums)
        
        for num in range(1, n):
            if counts[num] != 1:
                return False
        
        if counts[n] != 2:
            return False
        
        return True