from typing import List

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        # Find the maximum element in the array
        max_num = max(nums)
        
        # If the length of the array is not equal to max_num + 1, it cannot be a permutation of base[max_num]
        if len(nums) != max_num + 1:
            return False
        
        # Create a frequency dictionary to count the occurrences of each number
        freq = {}
        for num in nums:
            if num in freq:
                freq[num] += 1
            else:
                freq[num] = 1
        
        # Check if the frequency of each number matches the frequency in base[max_num]
        for i in range(1, max_num):
            if i not in freq or freq[i] != 1:
                return False
        
        # Check if the frequency of max_num is 2
        if max_num not in freq or freq[max_num] != 2:
            return False
        
        # If all checks pass, the array is a permutation of base[max_num]
        return True