class Solution:
    def calculateScore(self, s: str) -> int:
        def mirror(char):
            return chr(ord('z') - (ord(char) - ord('a')))

        n = len(s)
        score = 0
        marked = [False] * n

        for i in range(n):
            if not marked[i]:
                for j in range(i - 1, -1, -1):
                    if not marked[j] and s[j] == mirror(s[i]):
                        marked[i] = True
                        marked[j] = True
                        score += i - j
                        break
        return score