class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        arr1_strs = [str(x) for x in arr1]
        arr2_strs = [str(y) for y in arr2]
        
        # Calculate the maximum possible prefix length
        max_len = max(len(s) for s in arr1_strs + arr2_strs) if arr1_strs + arr2_strs else 0
        
        for L in range(max_len, 0, -1):
            s1 = set()
            for s in arr1_strs:
                if len(s) >= L:
                    s1.add(s[:L])
            
            s2 = set()
            for s in arr2_strs:
                if len(s) >= L:
                    s2.add(s[:L])
            
            if s1 & s2:
                return L
        
        return 0