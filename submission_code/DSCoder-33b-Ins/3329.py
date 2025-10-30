class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        def commonPrefix(a, b):
            return len(str(a)) if str(a) == str(b) else next((i for i, (x, y) in enumerate(zip(str(a), str(b))) if x != y), len(str(a)))

        return max(commonPrefix(a, b) for a in arr1 for b in arr2)