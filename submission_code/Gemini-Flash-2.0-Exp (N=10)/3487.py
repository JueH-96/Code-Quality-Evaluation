class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        m = len(pattern)
        
        def is_subsequence(s, p):
            i = 0
            j = 0
            while i < len(s) and j < len(p):
                if s[i] == p[j]:
                    j += 1
                i += 1
            return j == len(p)
        
        
        max_ops = 0
        for i in range(1 << len(targetIndices)):
            temp_source = list(source)
            ops = 0
            indices_to_remove = []
            for j in range(len(targetIndices)):
                if (i >> j) & 1:
                    indices_to_remove.append(targetIndices[j])
                    ops += 1
            
            indices_to_remove.sort(reverse=True)
            
            for index in indices_to_remove:
                temp_source.pop(index)
            
            if is_subsequence("".join(temp_source), pattern):
                max_ops = max(max_ops, ops)
        
        return max_ops