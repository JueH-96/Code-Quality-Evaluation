class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        def char_distance(c1, c2):
            # Calculate the cyclic distance between two characters
            diff = abs(ord(c1) - ord(c2))
            return min(diff, 26 - diff)
        
        n = len(s)
        result = list(s)  # Convert to list for easy character replacement
        
        for i in range(n):
            original_char = s[i]
            
            # Try replacing with each letter from 'a' to 'z'
            # Stop at the smallest one we can afford
            for char in "abcdefghijklmnopqrstuvwxyz":
                dist = char_distance(original_char, char)
                if dist <= k:
                    result[i] = char
                    k -= dist
                    break
        
        return ''.join(result)