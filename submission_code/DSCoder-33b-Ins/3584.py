class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        i, j, n, m = 0, 0, len(word1), len(word2)
        res = []
        while i < n and j < m:
            if word1[i] == word2[j]:
                res.append(i)
                j += 1
            i += 1
        if j < m:
            return []
        i = res[0]
        while i < n and len(res) < m:
            if word1[i] != word2[j-1]:
                res.append(i)
            i += 1
        return res if len(res) == m else []