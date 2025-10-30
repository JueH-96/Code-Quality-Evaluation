from sortedcontainers import SortedList

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        patternIdx, res = 0, 0
        targetIdx = SortedList(targetIndices)
        for i, c in enumerate(source):
            if patternIdx < len(pattern) and pattern[patternIdx] == c:
                if targetIdx and i == targetIdx[0]:
                    targetIdx.pop(0);
                    patternIdx += 1
                else:
                    res += 1
        return res + len(targetIdx)