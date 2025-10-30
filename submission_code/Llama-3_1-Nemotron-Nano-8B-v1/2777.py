from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix_set = set()
        prefix_counts = []
        for num in nums:
            prefix_set.add(num)
            prefix_counts.append(len(prefix_set))
        
        # Compute suffix_sets
        suffix_sets = [set() for _ in range(n + 1)]  # suffix_sets[n] is empty
        for i in range(n - 1, -1, -1):
            suffix_sets[i] = {nums[i]} | suffix_sets[i + 1]
        
        # Calculate the result
        result = []
        for i in range(n):
            suffix_size = len(suffix_sets[i + 1])
            result.append(prefix_counts[i] - suffix_size)
        
        return result