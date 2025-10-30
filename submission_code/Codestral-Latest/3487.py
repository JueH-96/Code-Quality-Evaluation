class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        def is_subsequence(s, p):
            it = iter(s)
            return all(char in it for char in p)

        left, right = 0, len(targetIndices)
        while left < right:
            mid = (left + right + 1) // 2
            temp_source = list(source)
            for i in range(mid):
                temp_source[targetIndices[i]] = ''
            if is_subsequence(''.join(temp_source), pattern):
                left = mid
            else:
                right = mid - 1
        return left