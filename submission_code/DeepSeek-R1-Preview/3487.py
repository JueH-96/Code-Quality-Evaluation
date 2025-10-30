class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        targets = set(targetIndices)
        m = len(pattern)
        previous = [float('inf')] * (m + 1)
        previous[0] = 0
        
        for i in range(1, len(source) + 1):
            current = previous.copy()
            c = source[i-1]
            in_target = 1 if (i-1) in targets else 0
            for j in range(1, m + 1):
                if c == pattern[j-1]:
                    option2 = previous[j-1] + in_target
                    if option2 < current[j]:
                        current[j] = option2
            previous = current
        
        minimal_overlap = previous[m]
        if minimal_overlap == float('inf'):
            return 0
        return len(targetIndices) - minimal_overlap