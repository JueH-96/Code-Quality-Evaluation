class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        i = 0
        while i < len(arr1) and i < len(arr2):
            if arr1[i] != arr2[i]:
                break
            i += 1
        return i