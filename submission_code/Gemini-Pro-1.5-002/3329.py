class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        max_len = 0
        for num1 in arr1:
            for num2 in arr2:
                s1 = str(num1)
                s2 = str(num2)
                i = 0
                while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
                    i += 1
                max_len = max(max_len, i)
        return max_len