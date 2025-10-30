class Solution:
    def calculateScore(self, s: str) -> int:
        n = len(s)
        marked = [False] * n
        score = 0

        def mirror(char):
            return chr(ord('z') - (ord(char) - ord('a')))

        for i in range(n):
            if marked[i]:
                continue
            
            closest_j = -1
            for j in range(i - 1, -1, -1):
                if not marked[j] and mirror(s[i]) == s[j]:
                    closest_j = j
                    break
            
            if closest_j != -1:
                marked[i] = True
                marked[closest_j] = True
                score += i - closest_j
        
        return score