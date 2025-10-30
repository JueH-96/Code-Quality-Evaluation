class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Convert all numbers in arr1 and arr2 to strings
        str_arr1 = [str(x) for x in arr1]
        str_arr2 = [str(x) for x in arr2]
        
        # Create a set of all possible prefixes for arr1
        prefix_set = set()
        for num in str_arr1:
            for i in range(1, len(num) + 1):
                prefix_set.add(num[:i])
        
        max_length = 0
        # Check for each number in arr2 if any prefix is in the prefix_set
        for num in str_arr2:
            for i in range(1, len(num) + 1):
                prefix = num[:i]
                if prefix in prefix_set:
                    if len(prefix) > max_length:
                        max_length = len(prefix)
        
        return max_length