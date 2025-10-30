class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        arr1_str = [str(num) for num in arr1]
        arr2_str = [str(num) for num in arr2]
        max_length = 0
        for l in range(1, 10):
            prefixes1 = set(s[:l] for s in arr1_str if len(s) >= l)
            prefixes2 = set(s[:l] for s in arr2_str if len(s) >= l)
            if prefixes1 & prefixes2:
                max_length = l
        return max_length