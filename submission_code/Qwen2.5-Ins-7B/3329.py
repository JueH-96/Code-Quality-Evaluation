class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        def get_prefixes(n):
            prefixes = set()
            for i in range(len(str(n))):
                prefixes.add(int(str(n)[:i+1]))
            return prefixes
        
        longest_prefix_length = 0
        arr1_prefixes = set()
        
        for num in arr1:
            arr1_prefixes.update(get_prefixes(num))
        
        for num in arr2:
            for prefix in get_prefixes(num):
                if prefix in arr1_prefixes:
                    longest_prefix_length = max(longest_prefix_length, len(str(prefix)))
        
        return longest_prefix_length