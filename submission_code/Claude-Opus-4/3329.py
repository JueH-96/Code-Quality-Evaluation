class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Store all prefixes of numbers in arr1
        prefix_set = set()
        
        # Generate all prefixes for each number in arr1
        for num in arr1:
            str_num = str(num)
            for i in range(1, len(str_num) + 1):
                prefix_set.add(str_num[:i])
        
        max_length = 0
        
        # For each number in arr2, check its prefixes against prefix_set
        for num in arr2:
            str_num = str(num)
            # Check prefixes from longest to shortest for efficiency
            for i in range(len(str_num), 0, -1):
                prefix = str_num[:i]
                if prefix in prefix_set:
                    max_length = max(max_length, len(prefix))
                    break  # Found the longest prefix for this number
        
        return max_length