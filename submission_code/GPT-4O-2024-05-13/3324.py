class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter
        
        # Count the frequency of each element in nums
        freq = Counter(nums)
        
        # Check if any element appears more than half the length of nums
        for count in freq.values():
            if count > len(nums) // 2:
                return False
        
        return True