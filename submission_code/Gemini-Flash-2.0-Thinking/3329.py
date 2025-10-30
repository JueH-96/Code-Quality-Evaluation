class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        max_len = 0

        for num1 in arr1:
            for num2 in arr2:
                str_num1 = str(num1)
                str_num2 = str(num2)
                current_len = 0
                min_len = min(len(str_num1), len(str_num2))

                for i in range(min_len):
                    if str_num1[i] == str_num2[i]:
                        current_len += 1
                    else:
                        break
                max_len = max(max_len, current_len)

        return max_len