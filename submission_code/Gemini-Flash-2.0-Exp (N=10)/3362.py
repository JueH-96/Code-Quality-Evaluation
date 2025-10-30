from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        uniqueness_array = []
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                uniqueness_array.append(len(set(subarray)))
        
        uniqueness_array.sort()
        
        mid = len(uniqueness_array) // 2
        
        return uniqueness_array[mid]