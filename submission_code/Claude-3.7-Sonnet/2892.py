class Solution:
    def isGood(self, nums: List[int]) -> bool:
        max_val = max(nums)
        
        # Check if the length of nums matches the expected length of base[max_val]
        if len(nums) != max_val + 1:
            return False
        
        # Count occurrences of integers in nums
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        # Check if nums contains integers from 1 to max_val-1 exactly once,
        # and max_val exactly twice
        for i in range(1, max_val):
            if count.get(i, 0) != 1:
                return False
        if count.get(max_val, 0) != 2:
            return False
        
        return True