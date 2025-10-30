class Solution:
    def isGood(self, nums: List[int]) -> bool:
        # The maximum value in nums
        max_val = max(nums)
        
        # For nums to be a permutation of base[max_val],
        # we need exactly (max_val + 1) elements
        if len(nums) != max_val + 1:
            return False
        
        # We expect 1 occurrence of each integer from 1 to max_val-1
        # and 2 occurrences of max_val
        from collections import Counter
        freq = Counter(nums)
        
        # Check frequency of each expected element
        for x in range(1, max_val):
            if freq[x] != 1:
                return False
        
        # Finally check the frequency of the maximum element
        return freq[max_val] == 2