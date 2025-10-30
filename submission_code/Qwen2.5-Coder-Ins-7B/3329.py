class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        max_length = 0
        for num1 in arr1:
            for num2 in arr2:
                common_prefix = 0
                str1, str2 = str(num1), str(num2)
                while common_prefix < len(str1) and common_prefix < len(str2) and str1[common_prefix] == str2[common_prefix]:
                    common_prefix += 1
                max_length = max(max_length, common_prefix)
        return max_length