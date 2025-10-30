from collections import defaultdict
from itertools import accumulate

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        last_occurrences = defaultdict(int)
        
        # Find the last occurrence index of each number
        for index, value in enumerate(nums):
            last_occurrences[value] = index
        
        nums_with_last_indices = sorted(last_occurrences.items())
        
        subarray_count = 1
        last_index = 0
        
        for value, last_index_of_value in nums_with_last_indices:
            if last_index <= last_index_of_value:
                last_index = last_index_of_value
                subarray_count += 1
                
        MOD = 10**9 + 7
        return pow(2, subarray_count - 1, MOD) if len(nums) > 1 else 1