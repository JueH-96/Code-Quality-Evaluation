class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        def common_prefix_length(a: str, b: str) -> int:
            min_len = min(len(a), len(b))
            for i in range(min_len):
                if a[i] != b[i]:
                    return i
            return min_len
        
        max_length = 0
        for num1 in arr1:
            str_num1 = str(num1)
            for num2 in arr2:
                str_num2 = str(num2)
                max_length = max(max_length, common_prefix_length(str_num1, str_num2))
        
        return max_length