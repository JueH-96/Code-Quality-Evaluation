class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Generate all prefixes for arr1 elements
        prefixes = set()
        for num in arr1:
            s = str(num)
            for i in range(1, len(s) + 1):
                prefixes.add(s[:i])
        
        max_len = 0
        # Check each element in arr2 for the longest prefix present in arr1's prefixes
        for num in arr2:
            s = str(num)
            # Check from longest to shortest prefix
            for i in range(len(s), 0, -1):
                if s[:i] in prefixes:
                    if i > max_len:
                        max_len = i
                    break  # Found the longest prefix for this element, move to next
        return max_len