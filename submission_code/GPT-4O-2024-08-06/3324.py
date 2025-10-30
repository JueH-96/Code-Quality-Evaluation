class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter
        
        # Count the frequency of each element in nums
        freq = Counter(nums)
        
        # Calculate the number of distinct elements
        distinct_count = len(freq)
        
        # The length of each split array
        half_length = len(nums) // 2
        
        # If the number of distinct elements is at least half the length of nums,
        # then it is possible to split the array into two parts with distinct elements.
        if distinct_count >= half_length:
            return True
        
        # Otherwise, it's not possible
        return False