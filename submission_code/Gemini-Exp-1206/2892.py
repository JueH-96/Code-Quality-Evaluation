class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 0:
            return False
        
        max_val = max(nums)
        
        if n != max_val + 1:
            return False
        
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        for i in range(1, max_val):
            if counts.get(i, 0) != 1:
                return False
        
        if counts.get(max_val, 0) != 2:
            return False
        
        return True