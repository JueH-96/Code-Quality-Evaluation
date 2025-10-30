class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        def common_prefix_length(x, y):
            length = 0
            for a, b in zip(str(x), str(y)):
                if a == b:
                    length += 1
                else:
                    break
            return length

        max_length = 0
        for x in arr1:
            for y in arr2:
                max_length = max(max_length, common_prefix_length(x, y))
        return max_length