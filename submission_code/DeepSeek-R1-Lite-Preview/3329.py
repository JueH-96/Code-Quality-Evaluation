class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Function to generate prefix sets for an array
        def get_prefix_sets(arr):
            prefix_sets = [set() for _ in range(9)]  # Index 1 to 8
            for num in arr:
                s = str(num)
                for L in range(1, len(s) + 1):
                    prefix = s[:L]
                    prefix_sets[L].add(prefix)
            return prefix_sets
        
        # Get prefix sets for both arrays
        prefix_sets_arr1 = get_prefix_sets(arr1)
        prefix_sets_arr2 = get_prefix_sets(arr2)
        
        # Check from the longest possible prefix down to 1
        for L in range(8, 0, -1):
            if prefix_sets_arr1[L] & prefix_sets_arr2[L]:
                return L
        return 0