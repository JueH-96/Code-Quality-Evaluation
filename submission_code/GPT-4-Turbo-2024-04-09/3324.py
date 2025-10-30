class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter
        
        # Count the frequency of each number in nums
        freq = Counter(nums)
        
        # The maximum frequency of any number should not exceed half the length of nums
        # because we need to split the array into two parts with distinct elements
        if max(freq.values()) > len(nums) // 2:
            return False
        
        # If all frequencies are less than or equal to half the length of nums,
        # we can always split the array into two parts with distinct elements
        return True