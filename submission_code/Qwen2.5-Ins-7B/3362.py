from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        def get_uniqueness_array(nums):
            n = len(nums)
            uniqueness = set()
            for i in range(n):
                for j in range(i, n):
                    uniqueness.add(len(set(nums[i:j+1])))
            return sorted(list(uniqueness))
        
        uniqueness_array = get_uniqueness_array(nums)
        n = len(uniqueness_array)
        median = uniqueness_array[n // 2] if n % 2 != 0 else uniqueness_array[(n // 2) - 1]
        return median