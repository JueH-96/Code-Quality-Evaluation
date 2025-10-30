class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        def common_prefix_length(x, y):
            x_str = str(x)
            y_str = str(y)
            min_len = min(len(x_str), len(y_str))
            for i in range(min_len):
                if x_str[i] != y_str[i]:
                    return i
            return min_len

        max_prefix_length = 0
        for x in arr1:
            for y in arr2:
                max_prefix_length = max(max_prefix_length, common_prefix_length(x, y))

        return max_prefix_length