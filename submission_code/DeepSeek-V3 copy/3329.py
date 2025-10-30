class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Function to get all prefixes of a number
        def get_prefixes(num):
            s = str(num)
            prefixes = set()
            for i in range(1, len(s)+1):
                prefixes.add(s[:i])
            return prefixes
        
        # Collect all prefixes from arr1
        prefixes1 = set()
        for num in arr1:
            prefixes1.update(get_prefixes(num))
        
        # Collect all prefixes from arr2
        prefixes2 = set()
        for num in arr2:
            prefixes2.update(get_prefixes(num))
        
        # Find common prefixes
        common = prefixes1.intersection(prefixes2)
        
        # Find the length of the longest common prefix
        max_len = 0
        for prefix in common:
            if len(prefix) > max_len:
                max_len = len(prefix)
        
        return max_len