from collections import Counter
from math import ceil

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        num_counts = Counter(nums)
        operations = 0
        
        for count in num_counts.values():
            if count == 1:
                return -1
            operations += ceil(count / 3)
            
        return operations