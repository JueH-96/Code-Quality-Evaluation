class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n = len(word1)
        m = len(word2)
        ans = []
        diff = 0
        i = 0
        j = 0
        while i < n and j < m:
            if word1[i] == word2[j]:
                ans.append(i)
                j += 1
            elif diff == 0:
                ans.append(i)
                diff += 1
                j += 1
            i += 1

        if j < m and diff == 0:
            while i < n and j < m:
                if word1[i] == word2[j]:
                    ans.append(i)
                    j += 1
                i += 1

        if len(ans) != m:
            return []

        
        res = []
        diff = 0
        i = 0
        j = 0

        while i < n and j < m:
            if word1[i] == word2[j]:
                res.append(i)
                j += 1
            elif diff == 0 and j < m:
                diff = 1
                res.append(i)
                j += 1
            i += 1
        
        if j < m and diff == 0:
            while i < n and j < m:
                if word1[i] == word2[j]:
                    res.append(i)
                    j += 1
                i += 1

        if len(res) == m:
            if len(ans) == 0 or res < ans:
                return res
            else:
                return ans
        else:
            return []