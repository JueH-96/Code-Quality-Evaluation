class Solution:
    def calculateScore(self, s: str) -> int:
        n = len(s)
        marked = [False] * n
        score = 0

        def get_mirror(char):
            return chr(ord('z') - (ord(char) - ord('a')))

        for i in range(n):
            if not marked[i]:
                mirror_char = get_mirror(s[i])
                closest_j = -1
                for j in range(i - 1, -1, -1):
                    if not marked[j] and s[j] == mirror_char:
                        closest_j = j
                        break

                if closest_j != -1:
                    marked[i] = True
                    marked[closest_j] = True
                    score += i - closest_j

        return score