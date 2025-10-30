class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Generate all prefixes of numbers in arr1
        prefixes = set()
        for num in arr1:
            s = str(num)
            for i in range(1, len(s) + 1):
                prefixes.add(s[:i])
        
        max_length = 0
        
        # For each number in arr2, check its prefixes
        for num in arr2:
            s = str(num)
            for i in range(1, len(s) + 1):
                prefix = s[:i]
                if prefix in prefixes:
                    max_length = max(max_length, i)
                else:
                    break  # No longer prefixes will be found
        
        return max_length