class Solution:
    def isGood(self, nums: list[int]) -> bool:
        # Determine candidate n: maximum element in nums
        n = max(nums)
        # Check if the length of nums equals n+1
        if len(nums) != n + 1:
            return False
        
        # Create a frequency counter for values in nums
        freq = {}
        for num in nums:
            # All numbers must be between 1 and n
            if num < 1 or num > n:
                return False
            freq[num] = freq.get(num, 0) + 1
        
        # Check numbers from 1 to n-1 each appear exactly once
        for i in range(1, n):
            if freq.get(i, 0) != 1:
                return False
        
        # Check that n appears exactly twice
        if freq.get(n, 0) != 2:
            return False
        
        return True