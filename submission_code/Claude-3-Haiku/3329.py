class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        if not arr1 or not arr2:
            return 0
        
        min_len = min(len(str(x)) for x in arr1 + arr2)
        
        for i in range(min_len):
            prefix = str(arr1[0])[0:i+1]
            if all(str(x).startswith(prefix) for x in arr1 + arr2):
                continue
            else:
                return i
        
        return min_len