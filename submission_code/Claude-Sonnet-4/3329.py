class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Store all prefixes from arr1
        prefixes = set()
        
        for num in arr1:
            str_num = str(num)
            # Add all prefixes of this number
            for i in range(1, len(str_num) + 1):
                prefixes.add(str_num[:i])
        
        max_length = 0
        
        # Check all prefixes of numbers in arr2
        for num in arr2:
            str_num = str(num)
            # Check all prefixes of this number
            for i in range(1, len(str_num) + 1):
                prefix = str_num[:i]
                if prefix in prefixes:
                    max_length = max(max_length, len(prefix))
        
        return max_length