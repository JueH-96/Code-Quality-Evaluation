class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        n = len(s)
        result = list(s)
        
        for i in range(n):
            min_dist = float('inf')
            min_char = result[i]
            
            for char in 'abcdefghijklmnopqrstuvwxyz':
                dist = 0
                for j in range(n):
                    if j == i:
                        dist += self.min_distance(s[j], char)
                    else:
                        dist += self.min_distance(s[j], result[j])
                
                if dist <= k and (dist, char) < (min_dist, min_char):
                    min_dist = dist
                    min_char = char
            
            result[i] = min_char
        
        return ''.join(result)
    
    def min_distance(self, s1: str, s2: str) -> int:
        return min(abs(ord(s1) - ord(s2)), 26 - abs(ord(s1) - ord(s2)))