class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        max_prefix_length = 0
        for x in arr1:
            for y in arr2:
                str_x = str(x)
                str_y = str(y)
                current_prefix_length = 0
                for i in range(min(len(str_x), len(str_y))):
                    if str_x[i] == str_y[i]:
                        current_prefix_length += 1
                    else:
                        break
                max_prefix_length = max(max_prefix_length, current_prefix_length)
        return max_prefix_length