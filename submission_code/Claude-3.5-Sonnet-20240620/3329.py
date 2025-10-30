class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        def get_prefixes(num):
            return set(str(num)[:i] for i in range(1, len(str(num)) + 1))
        
        prefixes1 = set()
        for num in arr1:
            prefixes1.update(get_prefixes(num))
        
        max_length = 0
        for num in arr2:
            for prefix in get_prefixes(num):
                if prefix in prefixes1:
                    max_length = max(max_length, len(prefix))
        
        return max_length