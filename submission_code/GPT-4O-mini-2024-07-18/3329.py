class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        def common_prefix_length(x: int, y: int) -> int:
            str_x, str_y = str(x), str(y)
            length = 0
            while length < min(len(str_x), len(str_y)) and str_x[length] == str_y[length]:
                length += 1
            return length
        
        max_length = 0
        for x in arr1:
            for y in arr2:
                max_length = max(max_length, common_prefix_length(x, y))
        
        return max_length