class Solution:
    def calculateScore(self, s: str) -> int:
        from collections import defaultdict
        
        score = 0
        
        # Mirror function: 'a' -> 'z', 'b' -> 'y', etc.
        def mirror(ch):
            return chr(ord('z') + ord('a') - ord(ch))
        
        # Dictionary to store lists of unmarked indices for each character
        char_indices = defaultdict(list)
        
        for i in range(len(s)):
            mirror_char = mirror(s[i])
            
            if char_indices[mirror_char]:
                # Pop the rightmost (closest) unmarked index
                j = char_indices[mirror_char].pop()
                score += i - j
            else:
                # Add current index to the list for its character
                char_indices[s[i]].append(i)
        
        return score