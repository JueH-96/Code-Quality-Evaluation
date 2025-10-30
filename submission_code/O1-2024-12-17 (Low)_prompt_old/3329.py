class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Create a set for all prefixes from arr1
        prefixes_arr1 = set()
        for num in arr1:
            s = str(num)
            for i in range(1, len(s)+1):
                prefixes_arr1.add(s[:i])
                
        # Create a set for all prefixes from arr2
        prefixes_arr2 = set()
        for num in arr2:
            s = str(num)
            for i in range(1, len(s)+1):
                prefixes_arr2.add(s[:i])
        
        # Find the intersection of both prefix sets
        common_prefixes = prefixes_arr1.intersection(prefixes_arr2)
        if not common_prefixes:
            return 0
        
        # Return the maximum length of any prefix in the intersection
        return max(len(p) for p in common_prefixes)