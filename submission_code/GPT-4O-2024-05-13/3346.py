class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        n = len(s)
        t = list(s)
        
        for i in range(n):
            if k == 0:
                break
            current_char = t[i]
            distance_to_a = (ord(current_char) - ord('a')) % 26
            if distance_to_a <= k:
                t[i] = 'a'
                k -= distance_to_a
            else:
                t[i] = chr(ord(current_char) - k)
                k = 0
        
        return ''.join(t)