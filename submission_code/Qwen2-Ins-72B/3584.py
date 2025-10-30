class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n, m = len(word1), len(word2)
        if m > n:
            return []
        
        def check(indices):
            for i in range(m):
                if word1[indices[i]] != word2[i]:
                    if i + 1 < m and word1[indices[i]] == word2[i + 1] and word1[indices[i + 1]] == word2[i]:
                        indices[i], indices[i + 1] = indices[i + 1], indices[i]
                    else:
                        return False
            return True
        
        indices = sorted([i for i in range(n) if word1[i] in set(word2)])
        if len(indices) < m:
            return []
        
        l, r = 0, len(indices) - m
        while l < r:
            mid = (l + r) // 2
            if check(indices[mid:mid + m]):
                r = mid
            else:
                l = mid + 1
        
        if check(indices[l:l + m]):
            return indices[l:l + m]
        else:
            return []