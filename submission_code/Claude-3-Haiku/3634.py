class Solution:
    def calculateScore(self, s: str) -> int:
        score = 0
        marked = [False] * len(s)
        
        for i in range(len(s)):
            mirror_index = -1
            min_distance = float('inf')
            
            for j in range(i):
                if not marked[j] and chr(ord('a') + ord('z') - ord(s[j])) == s[i]:
                    distance = i - j
                    if distance < min_distance:
                        min_distance = distance
                        mirror_index = j
            
            if mirror_index != -1:
                marked[i] = marked[mirror_index] = True
                score += min_distance
        
        return score