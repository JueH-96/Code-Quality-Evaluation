class Solution:
    def calculateScore(self, s: str) -> int:
        n = len(s)
        marked = [False] * n
        score = 0
        for i in range(n):
            if marked[i]:
                continue
            for j in range(i - 1, -1, -1):
                if not marked[j] and ord(s[i]) + ord(s[j]) == ord('a') + ord('z'):
                    marked[i] = True
                    marked[j] = True
                    score += i - j
                    break
        return score