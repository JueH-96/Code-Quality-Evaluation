class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = max(nums)
        
        # Check length
        if len(nums) != n + 1:
            return False
        
        # Count occurrences
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        # Check if each number from 1 to n-1 appears exactly once
        for i in range(1, n):
            if count.get(i, 0) != 1:
                return False
        
        # Check if n appears exactly twice
        if count.get(n, 0) != 2:
            return False
        
        return True