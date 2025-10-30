class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        set1 = set()
        for num in arr1:
            s = str(num)
            for i in range(1, len(s) + 1):
                set1.add(s[:i])
        
        set2 = set()
        for num in arr2:
            s = str(num)
            for i in range(1, len(s) + 1):
                set2.add(s[:i])
        
        common = set1 & set2
        if not common:
            return 0
        return max(len(prefix) for prefix in common)