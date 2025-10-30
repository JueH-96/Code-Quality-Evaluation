class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Store all prefixes of arr1
        prefixes = set()
        
        for num in arr1:
            str_num = str(num)
            for i in range(1, len(str_num) + 1):
                prefixes.add(str_num[:i])
        
        max_length = 0
        
        for num in arr2:
            str_num = str(num)
            # Check prefixes from longest to shortest
            for i in range(len(str_num), 0, -1):
                prefix = str_num[:i]
                if prefix in prefixes:
                    max_length = max(max_length, len(prefix))
                    break  # No need to check shorter prefixes
        
        return max_length