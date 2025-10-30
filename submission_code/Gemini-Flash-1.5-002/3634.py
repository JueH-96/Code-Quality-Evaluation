class Solution:
    def calculateScore(self, s: str) -> int:
        score = 0
        marked = [False] * len(s)
        mirror = {}
        for i in range(26):
            mirror[chr(ord('a') + i)] = chr(ord('z') - i)

        for i in range(len(s)):
            if marked[i]:
                continue
            best_j = -1
            for j in range(i - 1, -1, -1):
                if marked[j]:
                    continue
                if mirror[s[i]] == s[j]:
                    best_j = j
                    break
            if best_j != -1:
                score += i - best_j
                marked[i] = True
                marked[best_j] = True

        return score