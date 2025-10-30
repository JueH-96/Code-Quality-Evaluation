class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        def common_prefix_length(x, y):
            i = 0
            while i < min(len(x), len(y)):
                if x[i] != y[i]:
                    break
                i += 1
            return i
        
        arr1 = list(map(str, arr1))
        arr2 = list(map(str, arr2))
        pre = 0
        for s1 in arr1:
            for s2 in arr2:
                pre = max(pre, common_prefix_length(s1, s2))
        return pre