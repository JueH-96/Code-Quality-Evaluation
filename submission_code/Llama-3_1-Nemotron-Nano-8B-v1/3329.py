from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        def get_prefixes(nums):
            prefixes = set()
            for num in nums:
                s = str(num)
                for i in range(1, len(s) + 1):
                    prefixes.add(s[:i])
            return prefixes
        
        prefixes1 = get_prefixes(arr1)
        prefixes2 = get_prefixes(arr2)
        
        common = prefixes1.intersection(prefixes2)
        if not common:
            return 0
        return max(len(prefix) for prefix in common)