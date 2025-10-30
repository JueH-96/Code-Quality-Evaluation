class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        n = len(s)
        t = list(s)
        
        for i in range(n):
            if k <= 0:
                break
            
            current_char = t[i]
            target_char = 'a'
            
            # Calculate the distance to 'a'
            distance_to_a = (ord(current_char) - ord(target_char)) % 26
            
            if distance_to_a <= k:
                # Change the character to 'a'
                t[i] = target_char
                k -= distance_to_a
            else:
                # Change the character to the lexicographically smallest possible
                # character within the remaining k distance
                t[i] = chr(ord(current_char) - k)
                k = 0
        
        return ''.join(t)